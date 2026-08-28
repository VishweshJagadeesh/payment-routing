"""
Intelligent Payment Routing & Optimization System
====================================================

A simulation of an ML-driven, fault-tolerant payment routing engine, of the
kind used by payment platforms (Stripe, Adyen, Braintree) to decide which
downstream payment gateway/processor should handle each transaction.

Everything in this file is a SIMULATION. No real payment processing, card
data, or financial settlement occurs anywhere in this program. See README.md
for a full description of what is simulated vs. not implemented.

Run:
    python payment_router.py

Author: Portfolio project for Fintech / Payments SDE & Finance-ML roles.
"""

from __future__ import annotations

import random
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =====================================================================
# 1. DOMAIN CONSTANTS & GROUND-TRUTH GATEWAY PROFILES
# =====================================================================
#
# These profiles define a hidden "ground truth" performance surface for
# each simulated gateway. Both the historical-data generator and the
# real-time simulator sample from this same surface, so the ML model
# trained on historical data is learning a genuinely predictive
# relationship rather than pure noise. Every number below is a synthetic
# authoring choice for this simulation, not a real vendor statistic.

COUNTRIES = ["US", "GB", "IN", "DE", "BR", "AU"]
PAYMENT_METHODS = ["card", "wallet", "bank_transfer", "bnpl"]
GATEWAYS = ["gateway_a", "gateway_b", "gateway_c"]

# Base authorization success rate per gateway (before modifiers).
GATEWAY_BASE_SUCCESS = {
    "gateway_a": 0.93,
    "gateway_b": 0.88,
    "gateway_c": 0.90,
}

# Per-country multiplicative modifier on success probability, representing
# regional acquiring strength (e.g. a gateway with strong EU acquiring
# banks performs better in DE/GB than in BR).
GATEWAY_COUNTRY_MODIFIER = {
    "gateway_a": {"US": 1.00, "GB": 0.98, "IN": 0.90, "DE": 0.97, "BR": 0.80, "AU": 0.96},
    "gateway_b": {"US": 0.95, "GB": 0.94, "IN": 0.97, "DE": 0.90, "BR": 0.93, "AU": 0.90},
    "gateway_c": {"US": 0.97, "GB": 0.93, "IN": 0.85, "DE": 0.95, "BR": 0.88, "AU": 0.94},
}

# Per-payment-method multiplicative modifier.
GATEWAY_METHOD_MODIFIER = {
    "gateway_a": {"card": 1.00, "wallet": 0.97, "bank_transfer": 0.90, "bnpl": 0.85},
    "gateway_b": {"card": 0.96, "wallet": 1.00, "bank_transfer": 0.93, "bnpl": 0.90},
    "gateway_c": {"card": 0.98, "wallet": 0.95, "bank_transfer": 0.97, "bnpl": 0.80},
}

# Base latency (ms, mean of a log-normal-ish distribution) and processing
# cost model (fixed fee + percentage of amount) per gateway.
GATEWAY_LATENCY_MS = {"gateway_a": 220, "gateway_b": 340, "gateway_c": 180}
GATEWAY_COST_PCT = {"gateway_a": 0.029, "gateway_b": 0.024, "gateway_c": 0.031}
GATEWAY_COST_FIXED = {"gateway_a": 0.30, "gateway_b": 0.10, "gateway_c": 0.25}


def true_success_probability(
    gateway: str, country: str, payment_method: str, amount: float, hour: int, load_factor: float = 1.0
) -> float:
    """The hidden ground-truth P(success) surface used to *sample* outcomes.

    This function is intentionally analogous to, but not identical to, what
    the ML model must learn from noisy historical samples. It encodes:
      - a gateway/country/payment-method base surface,
      - a mild negative effect of very large transaction amounts (higher
        fraud-review friction on big-ticket transactions),
      - a mild time-of-day effect (overnight batch/maintenance windows),
      - a load effect (a gateway under heavy simulated load authorizes
        slightly worse, mimicking queueing/timeout pressure).
    """
    base = GATEWAY_BASE_SUCCESS[gateway]
    base *= GATEWAY_COUNTRY_MODIFIER[gateway][country]
    base *= GATEWAY_METHOD_MODIFIER[gateway][payment_method]

    # Large amounts are modestly harder to authorize (risk engines bite).
    amount_effect = -0.06 * max(0.0, (amount - 500) / 2000)

    # Overnight hours (00:00-05:00) see a small dip (maintenance windows).
    time_effect = -0.03 if hour < 5 else 0.0

    # Heavier simulated load degrades authorization slightly.
    load_effect = -0.10 * max(0.0, load_factor - 1.0)

    p = base + amount_effect + time_effect + load_effect
    return float(np.clip(p, 0.02, 0.995))


def true_latency_ms(gateway: str, amount: float, load_factor: float) -> float:
    base = GATEWAY_LATENCY_MS[gateway]
    amount_component = amount * 0.02
    load_component = base * 0.6 * max(0.0, load_factor - 1.0)
    noise = np.random.gamma(shape=2.0, scale=base / 8)
    return max(15.0, base + amount_component + load_component + noise)


def true_processing_cost(gateway: str, amount: float) -> float:
    return round(amount * GATEWAY_COST_PCT[gateway] + GATEWAY_COST_FIXED[gateway], 4)


# =====================================================================
# 2. STATE MACHINE & ENUMS
# =====================================================================

class TransactionState(Enum):
    INITIATED = "INITIATED"
    AUTHORIZED = "AUTHORIZED"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
    COMPLETED = "COMPLETED"


# Allowed transitions, enforced by TransactionRecord.transition().
_ALLOWED_TRANSITIONS = {
    TransactionState.INITIATED: {TransactionState.AUTHORIZED, TransactionState.FAILED, TransactionState.RETRYING},
    TransactionState.RETRYING: {TransactionState.AUTHORIZED, TransactionState.FAILED, TransactionState.RETRYING},
    TransactionState.AUTHORIZED: {TransactionState.COMPLETED, TransactionState.FAILED},
    TransactionState.FAILED: set(),      # terminal
    TransactionState.COMPLETED: set(),   # terminal
}


class GatewayStatus(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"


class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


# =====================================================================
# 3. REQUEST / RECORD DATA MODELS
# =====================================================================

@dataclass
class PaymentRequest:
    merchant_id: str
    amount: float
    currency: str
    country: str
    payment_method: str
    idempotency_key: str
    transaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    def validate(self) -> Tuple[bool, Optional[str]]:
        if self.amount is None or self.amount <= 0:
            return False, "amount must be positive"
        if self.currency is None or len(self.currency) != 3:
            return False, "currency must be a 3-letter ISO code"
        if self.country not in COUNTRIES:
            return False, f"unsupported country '{self.country}'"
        if self.payment_method not in PAYMENT_METHODS:
            return False, f"unsupported payment method '{self.payment_method}'"
        if not self.idempotency_key:
            return False, "idempotency_key is required"
        return True, None


@dataclass
class AttemptRecord:
    gateway: str
    success: bool
    latency_ms: float
    cost: float
    reason: str  # "authorized" | "declined" | "timeout" | "circuit_open" | "unhealthy"


@dataclass
class TransactionRecord:
    request: PaymentRequest
    state: TransactionState = TransactionState.INITIATED
    attempts: List[AttemptRecord] = field(default_factory=list)
    final_gateway: Optional[str] = None
    total_latency_ms: float = 0.0
    total_cost: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    from_idempotency_cache: bool = False

    def transition(self, new_state: TransactionState) -> None:
        allowed = _ALLOWED_TRANSITIONS[self.state]
        if new_state not in allowed:
            raise ValueError(f"Illegal transition {self.state.value} -> {new_state.value}")
        self.state = new_state
        if new_state in (TransactionState.COMPLETED, TransactionState.FAILED):
            self.completed_at = datetime.now()

    @property
    def success(self) -> bool:
        return self.state == TransactionState.COMPLETED


# =====================================================================
# 4. IDEMPOTENCY STORE
# =====================================================================

class IdempotencyStore:
    """Ensures the same client-supplied idempotency_key never results in
    two successful payment records, mirroring how real payment APIs
    (e.g. Stripe's Idempotency-Key header) prevent duplicate charges on
    client retry."""

    def __init__(self) -> None:
        self._store: Dict[str, TransactionRecord] = {}

    def get(self, key: str) -> Optional[TransactionRecord]:
        return self._store.get(key)

    def put(self, key: str, record: TransactionRecord) -> None:
        self._store[key] = record

    def __len__(self) -> int:
        return len(self._store)


# =====================================================================
# 5. CIRCUIT BREAKER
# =====================================================================

class CircuitBreaker:
    """Per-gateway circuit breaker (classic CLOSED -> OPEN -> HALF_OPEN
    state machine) that stops routing traffic to a gateway that is
    failing repeatedly, and probes recovery after a cooldown window.

    The cooldown is measured on a *logical* clock (a monotonically
    increasing counter of routed requests, supplied by PaymentRouter)
    rather than wall-clock time. In a live production system wall-clock
    time is the natural choice; in this simulation thousands of
    transactions are routed in a few milliseconds of real time, so a
    wall-clock cooldown would never elapse and an opened circuit would
    stay open (and effectively lock out that gateway) for the rest of
    the run. A request-count-based clock keeps the recovery behavior
    meaningful regardless of how fast the simulation executes."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 40.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state: Dict[str, CircuitState] = {gw: CircuitState.CLOSED for gw in GATEWAYS}
        self._consecutive_failures: Dict[str, int] = {gw: 0 for gw in GATEWAYS}
        self._opened_at: Dict[str, float] = {}

    def allow(self, gateway: str, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.monotonic()
        state = self._state[gateway]
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.OPEN:
            if now - self._opened_at.get(gateway, 0) >= self.recovery_timeout:
                self._state[gateway] = CircuitState.HALF_OPEN
                return True
            return False
        if state == CircuitState.HALF_OPEN:
            # Allow a single trial request through in HALF_OPEN.
            return True
        return True

    def record_result(self, gateway: str, success: bool, now: Optional[float] = None) -> None:
        now = now if now is not None else time.monotonic()
        if success:
            self._consecutive_failures[gateway] = 0
            self._state[gateway] = CircuitState.CLOSED
        else:
            self._consecutive_failures[gateway] += 1
            if self._state[gateway] == CircuitState.HALF_OPEN:
                # Trial failed -> re-open immediately.
                self._state[gateway] = CircuitState.OPEN
                self._opened_at[gateway] = now
            elif self._consecutive_failures[gateway] >= self.failure_threshold:
                self._state[gateway] = CircuitState.OPEN
                self._opened_at[gateway] = now

    def state_of(self, gateway: str) -> CircuitState:
        return self._state[gateway]

    def force_open(self, gateway: str, now: Optional[float] = None) -> None:
        now = now if now is not None else time.monotonic()
        self._state[gateway] = CircuitState.OPEN
        self._opened_at[gateway] = now
        self._consecutive_failures[gateway] = self.failure_threshold


def exponential_backoff_delay(attempt: int, base_ms: float = 50.0, cap_ms: float = 2000.0) -> float:
    """Exponential backoff with full jitter (AWS-style), returned in ms."""
    ceiling = min(cap_ms, base_ms * (2 ** attempt))
    return random.uniform(0, ceiling)


# =====================================================================
# 6. GATEWAY HEALTH MONITOR
# =====================================================================

class GatewayHealthMonitor:
    """Tracks a rolling window of recent outcomes per gateway and derives
    a live HEALTHY / DEGRADED / OFFLINE status, plus a `load_factor` used
    to feed back into the ground-truth simulation (more in-flight
    "load" -> slightly worse real-world performance, mimicking
    queueing)."""

    def __init__(self, window: int = 50):
        self.window = window
        self._outcomes: Dict[str, deque] = {gw: deque(maxlen=window) for gw in GATEWAYS}
        self._latencies: Dict[str, deque] = {gw: deque(maxlen=window) for gw in GATEWAYS}
        self._inflight: Dict[str, int] = {gw: 0 for gw in GATEWAYS}
        self._manual_status: Dict[str, Optional[GatewayStatus]] = {gw: None for gw in GATEWAYS}

    def begin_request(self, gateway: str) -> None:
        self._inflight[gateway] += 1

    def end_request(self, gateway: str) -> None:
        self._inflight[gateway] = max(0, self._inflight[gateway] - 1)

    def record(self, gateway: str, success: bool, latency_ms: float) -> None:
        self._outcomes[gateway].append(1 if success else 0)
        self._latencies[gateway].append(latency_ms)

    def rolling_success_rate(self, gateway: str) -> float:
        outcomes = self._outcomes[gateway]
        if not outcomes:
            return 1.0  # optimistic prior before any data
        return sum(outcomes) / len(outcomes)

    def rolling_avg_latency(self, gateway: str) -> float:
        lat = self._latencies[gateway]
        if not lat:
            return GATEWAY_LATENCY_MS[gateway]
        return sum(lat) / len(lat)

    def load_factor(self, gateway: str) -> float:
        # Normalize in-flight count into a load multiplier, e.g. 5
        # concurrent in-flight requests ~= 1.5x nominal load.
        return 1.0 + self._inflight[gateway] / 10.0

    def set_manual_status(self, gateway: str, status: Optional[GatewayStatus]) -> None:
        """Allows the simulation to force a gateway into a given status,
        e.g. to demonstrate an outage scenario."""
        self._manual_status[gateway] = status

    def status_of(self, gateway: str) -> GatewayStatus:
        # OFFLINE is intentionally never inferred automatically from noisy
        # rolling statistics: doing so can create a self-reinforcing lockout
        # (an OFFLINE gateway stops receiving traffic, so it can never
        # collect the data needed to recover). OFFLINE is instead a state
        # that is either forced by an operator/simulation (see
        # `set_manual_status`, used in the outage demo below) or reached
        # algorithmically via the CircuitBreaker, which *does* have a
        # built-in recovery path (HALF_OPEN probing). Automatic health
        # scoring here is limited to HEALTHY / DEGRADED.
        if self._manual_status[gateway] is not None:
            return self._manual_status[gateway]
        rate = self.rolling_success_rate(gateway)
        if len(self._outcomes[gateway]) < 10:
            return GatewayStatus.HEALTHY
        # Threshold is set well below typical per-gateway authorization
        # rates (~0.70-0.90 in this simulation) so ordinary binomial noise
        # in a rolling window doesn't cause spurious flapping; DEGRADED is
        # meant to catch a genuine, sustained drop in performance.
        if rate >= 0.55:
            return GatewayStatus.HEALTHY
        return GatewayStatus.DEGRADED


# =====================================================================
# 7. SYNTHETIC HISTORICAL DATA GENERATION
# =====================================================================

def generate_historical_data(n: int = 20000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generates n synthetic historical transactions, each routed to a
    randomly-chosen gateway (as if by a naive round-robin router), with
    outcomes sampled from the ground-truth surface above. This is the
    'training data' the ML routing model learns from."""
    rng = np.random.default_rng(seed)
    start = datetime(2025, 1, 1)

    merchants = [f"merchant_{i:03d}" for i in range(60)]
    merchant_amount_scale = {m: rng.lognormal(mean=3.5, sigma=0.6) for m in merchants}

    rows = []
    for i in range(n):
        merchant_id = merchants[rng.integers(0, len(merchants))]
        country = COUNTRIES[rng.integers(0, len(COUNTRIES))]
        payment_method = PAYMENT_METHODS[rng.integers(0, len(PAYMENT_METHODS))]
        gateway = GATEWAYS[rng.integers(0, len(GATEWAYS))]

        amount = float(np.clip(rng.lognormal(mean=3.2, sigma=1.0) * (merchant_amount_scale[merchant_id] / 30), 1, 9000))
        ts = start + timedelta(minutes=int(rng.integers(0, 60 * 24 * 120)))
        hour = ts.hour

        # Simulated historical gateway load: mildly time-of-day dependent
        # (peak hours 9-21 see more traffic / load).
        load_factor = 1.0 + (0.4 if 9 <= hour <= 21 else 0.0) + float(rng.normal(0, 0.1))
        load_factor = max(0.5, load_factor)

        p_success = true_success_probability(gateway, country, payment_method, amount, hour, load_factor)
        success = bool(rng.random() < p_success)
        latency = true_latency_ms(gateway, amount, load_factor)
        cost = true_processing_cost(gateway, amount)

        rows.append(
            {
                "transaction_id": str(uuid.uuid4()),
                "merchant_id": merchant_id,
                "amount": round(amount, 2),
                "country": country,
                "currency": {"US": "USD", "GB": "GBP", "IN": "INR", "DE": "EUR", "BR": "BRL", "AU": "AUD"}[country],
                "payment_method": payment_method,
                "gateway": gateway,
                "timestamp": ts,
                "success": success,
                "latency_ms": round(latency, 2),
                "processing_cost": cost,
            }
        )

    df = pd.DataFrame(rows)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    return df


# =====================================================================
# 8. ML ROUTING MODEL: P(success | transaction, gateway)
# =====================================================================

CATEGORICAL_FEATURES = ["gateway", "country", "payment_method"]
NUMERIC_FEATURES = ["amount", "hour", "day_of_week"]


def build_pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",
    )
    return Pipeline(steps=[("pre", pre), ("clf", estimator)])


@dataclass
class ModelMetrics:
    name: str
    roc_auc: float
    pr_auc: float
    auth_rate_at_threshold: float
    calibration_error: float


class RoutingModel:
    """Trains and compares two classifiers that estimate
    P(payment_success | transaction, gateway), then exposes the better
    one (by ROC-AUC) as `predict_proba` for use by the ML router."""

    def __init__(self):
        self.pipelines: Dict[str, Pipeline] = {}
        self.metrics: Dict[str, ModelMetrics] = {}
        self.best_model_name: Optional[str] = None

    def train(self, df: pd.DataFrame) -> Dict[str, ModelMetrics]:
        feature_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES
        X = df[feature_cols]
        y = df["success"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
        )

        candidates = {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
        }

        for name, estimator in candidates.items():
            pipe = build_pipeline(estimator)
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:, 1]

            roc = roc_auc_score(y_test, proba)
            pr = average_precision_score(y_test, proba)

            # Authorization rate if we approve every transaction whose
            # predicted P(success) exceeds 0.5 (a simple operating point).
            approved = proba >= 0.5
            auth_rate = float(y_test[approved].mean()) if approved.sum() > 0 else 0.0

            # Calibration error: mean absolute gap between predicted-bucket
            # probability and observed frequency, across 10 bins.
            frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
            calib_err = float(np.mean(np.abs(frac_pos - mean_pred)))

            self.pipelines[name] = pipe
            self.metrics[name] = ModelMetrics(name, roc, pr, auth_rate, calib_err)

        self.best_model_name = max(self.metrics, key=lambda k: self.metrics[k].roc_auc)
        return self.metrics

    def predict_success_proba(self, gateway: str, country: str, payment_method: str,
                               amount: float, hour: int, day_of_week: int) -> float:
        row = pd.DataFrame([{
            "gateway": gateway, "country": country, "payment_method": payment_method,
            "amount": amount, "hour": hour, "day_of_week": day_of_week,
        }])
        pipe = self.pipelines[self.best_model_name]
        return float(pipe.predict_proba(row)[:, 1][0])


# =====================================================================
# 9. ROUTING OBJECTIVE (EXPECTED UTILITY)
# =====================================================================

@dataclass
class RoutingWeights:
    """All weights in the routing objective are explicit and configurable
    here, rather than buried as unexplained magic numbers in the routing
    logic. Defaults below reflect one reasonable business stance (expected
    revenue is the primary driver, cost and latency are meaningful but
    secondary) -- tune per business priorities.

    utility = success_weight   * P(success) * amount
              - cost_weight    * processing_cost
              - latency_weight * (latency_ms / 1000)
    """
    success_weight: float = 1.0
    cost_weight: float = 1.0
    latency_weight: float = 0.05  # penalty per second of expected latency


def expected_utility(p_success: float, amount: float, cost: float, latency_ms: float,
                      weights: RoutingWeights) -> float:
    return (
        weights.success_weight * p_success * amount
        - weights.cost_weight * cost
        - weights.latency_weight * (latency_ms / 1000.0)
    )


# =====================================================================
# 10. ROUTING STRATEGIES
# =====================================================================

class RoutingStrategy:
    """Common interface: rank candidate gateways best-first for a request."""

    name = "base"

    def rank_candidates(self, request: PaymentRequest, candidates: List[str]) -> List[str]:
        raise NotImplementedError


class FixedGatewayRouter(RoutingStrategy):
    """Baseline #1: a single hard-coded gateway integration with no
    fallback -- representative of the common starting point for a
    payments platform (one processor integration, wired in directly).
    If the fixed gateway is unavailable or declines, there is nothing
    else to fall back to, which is the realistic cost of not having a
    routing layer at all."""

    name = "fixed"

    def __init__(self, fixed_gateway: str = "gateway_a"):
        self.fixed_gateway = fixed_gateway

    def rank_candidates(self, request: PaymentRequest, candidates: List[str]) -> List[str]:
        return [g for g in candidates if g == self.fixed_gateway]


class RuleBasedRouter(RoutingStrategy):
    """Baseline #2: a hand-written heuristic router. Picks the gateway
    with the best *historical* observed success rate for the request's
    (country, payment_method) segment, breaking ties by lower cost. This
    represents what many teams ship before investing in ML routing."""

    name = "rule_based"

    def __init__(self, historical_df: pd.DataFrame):
        agg = (
            historical_df.groupby(["country", "payment_method", "gateway"])["success"]
            .mean()
            .reset_index()
            .rename(columns={"success": "hist_success_rate"})
        )
        self._lookup: Dict[Tuple[str, str, str], float] = {
            (r.country, r.payment_method, r.gateway): r.hist_success_rate for r in agg.itertuples()
        }

    def rank_candidates(self, request: PaymentRequest, candidates: List[str]) -> List[str]:
        def score(gw: str) -> Tuple[float, float]:
            hist = self._lookup.get((request.country, request.payment_method, gw), 0.5)
            cost = true_processing_cost(gw, request.amount)
            return (-hist, cost)  # sort ascending: best success first, then lowest cost
        return sorted(candidates, key=score)


class MLRouter(RoutingStrategy):
    """Production candidate: scores every healthy candidate gateway with
    the trained RoutingModel and ranks by expected utility."""

    name = "ml_routing"

    def __init__(self, model: RoutingModel, weights: RoutingWeights):
        self.model = model
        self.weights = weights

    def rank_candidates(self, request: PaymentRequest, candidates: List[str]) -> List[str]:
        hour = request.timestamp.hour
        dow = request.timestamp.weekday()
        scored = []
        for gw in candidates:
            p = self.model.predict_success_proba(gw, request.country, request.payment_method,
                                                   request.amount, hour, dow)
            cost = true_processing_cost(gw, request.amount)
            latency_est = GATEWAY_LATENCY_MS[gw]
            u = expected_utility(p, request.amount, cost, latency_est, self.weights)
            scored.append((u, gw))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [gw for _, gw in scored]


# =====================================================================
# 11. PAYMENT PROCESSOR (simulated gateway call)
# =====================================================================

class PaymentProcessor:
    """Simulates actually calling a gateway: applies a request timeout,
    samples an outcome from the ground-truth surface (modulated by the
    gateway's *current* health/load), and returns an AttemptRecord.
    No real network call is made."""

    def __init__(self, health_monitor: GatewayHealthMonitor, timeout_ms: float = 3000.0):
        self.health_monitor = health_monitor
        self.timeout_ms = timeout_ms

    def attempt(self, gateway: str, request: PaymentRequest) -> AttemptRecord:
        self.health_monitor.begin_request(gateway)
        try:
            status = self.health_monitor.status_of(gateway)
            load = self.health_monitor.load_factor(gateway)

            if status == GatewayStatus.OFFLINE:
                # Offline gateways fail fast without consuming a full
                # simulated latency budget.
                return AttemptRecord(gateway, False, 5.0, 0.0, "unhealthy")

            hour = request.timestamp.hour
            degradation = 1.6 if status == GatewayStatus.DEGRADED else 1.0
            p_success = true_success_probability(
                gateway, request.country, request.payment_method, request.amount, hour, load * degradation
            )
            latency = true_latency_ms(gateway, request.amount, load * degradation)

            if latency > self.timeout_ms:
                return AttemptRecord(gateway, False, self.timeout_ms, 0.0, "timeout")

            success = random.random() < p_success
            cost = true_processing_cost(gateway, request.amount) if success else 0.0
            reason = "authorized" if success else "declined"
            record = AttemptRecord(gateway, success, latency, cost, reason)
            self.health_monitor.record(gateway, success, latency)
            return record
        finally:
            self.health_monitor.end_request(gateway)


# =====================================================================
# 12. PAYMENT ROUTER (orchestration)
# =====================================================================

@dataclass
class RouterConfig:
    max_retries: int = 2
    backoff_base_ms: float = 20.0
    backoff_cap_ms: float = 400.0
    simulate_sleep: bool = False  # real time.sleep for backoff (off during bulk sim)


class PaymentRouter:
    """Ties together validation, idempotency, health/circuit checks,
    candidate scoring, and retry/fallback into a single `route()` call
    that mirrors a production routing service's request lifecycle."""

    def __init__(self, strategy: RoutingStrategy, health_monitor: GatewayHealthMonitor,
                 circuit_breaker: CircuitBreaker, idempotency_store: IdempotencyStore,
                 processor: PaymentProcessor, config: RouterConfig):
        self.strategy = strategy
        self.health_monitor = health_monitor
        self.circuit_breaker = circuit_breaker
        self.idempotency_store = idempotency_store
        self.processor = processor
        self.config = config
        self._clock: float = 0.0  # logical clock; see CircuitBreaker docstring

    def _eligible_candidates(self) -> List[str]:
        eligible = []
        for gw in GATEWAYS:
            if self.health_monitor.status_of(gw) == GatewayStatus.OFFLINE:
                continue
            if not self.circuit_breaker.allow(gw, now=self._clock):
                continue
            eligible.append(gw)
        return eligible if eligible else list(GATEWAYS)  # last-resort: try everything

    def route(self, request: PaymentRequest) -> TransactionRecord:
        self._clock += 1.0  # advance logical clock once per routed request

        # Step 1: idempotency check happens before anything else.
        cached = self.idempotency_store.get(request.idempotency_key)
        if cached is not None:
            cached.from_idempotency_cache = True
            return cached

        record = TransactionRecord(request=request)

        # Step 2: validation.
        ok, reason = request.validate()
        if not ok:
            record.transition(TransactionState.FAILED)
            record.attempts.append(AttemptRecord("none", False, 0.0, 0.0, f"validation_error:{reason}"))
            self.idempotency_store.put(request.idempotency_key, record)
            return record

        # Step 3-6: health-filtered candidate list, scored and ranked.
        candidates = self._eligible_candidates()
        ranked = self.strategy.rank_candidates(request, candidates)

        attempt_no = 0
        while attempt_no <= self.config.max_retries and ranked:
            gateway = ranked.pop(0)
            attempt_no += 1

            if not self.circuit_breaker.allow(gateway, now=self._clock):
                record.attempts.append(AttemptRecord(gateway, False, 0.0, 0.0, "circuit_open"))
                continue

            outcome = self.processor.attempt(gateway, request)
            record.attempts.append(outcome)
            record.total_latency_ms += outcome.latency_ms
            self.circuit_breaker.record_result(gateway, outcome.success, now=self._clock)

            if outcome.success:
                record.final_gateway = gateway
                record.total_cost += outcome.cost
                record.transition(TransactionState.AUTHORIZED)
                record.transition(TransactionState.COMPLETED)
                break
            else:
                # Failed attempt: move to RETRYING if we still have budget
                # and fallback candidates, and back off exponentially
                # before the next attempt (skipped during bulk simulation
                # for speed; the delay is still accounted for in metrics).
                if attempt_no <= self.config.max_retries and ranked:
                    if record.state == TransactionState.INITIATED:
                        record.transition(TransactionState.RETRYING)
                    delay = exponential_backoff_delay(
                        attempt_no, self.config.backoff_base_ms, self.config.backoff_cap_ms
                    )
                    record.total_latency_ms += delay
                    if self.config.simulate_sleep:
                        time.sleep(delay / 1000.0)

        if record.state not in (TransactionState.COMPLETED, TransactionState.FAILED):
            # Exhausted retries/candidates without a successful attempt.
            record.transition(TransactionState.FAILED)

        self.idempotency_store.put(request.idempotency_key, record)
        return record


# =====================================================================
# 13. SYNTHETIC LIVE-TRAFFIC GENERATOR
# =====================================================================

def generate_live_requests(n: int, merchants: List[str], seed: int) -> List[PaymentRequest]:
    """Generates a stream of incoming PaymentRequest objects for the
    real-time simulation, independent of the historical training data."""
    rng = np.random.default_rng(seed)
    requests = []
    start = datetime(2025, 6, 1)
    for i in range(n):
        merchant_id = merchants[rng.integers(0, len(merchants))]
        country = COUNTRIES[rng.integers(0, len(COUNTRIES))]
        payment_method = PAYMENT_METHODS[rng.integers(0, len(PAYMENT_METHODS))]
        amount = float(np.clip(rng.lognormal(mean=3.2, sigma=1.0), 1, 9000))
        ts = start + timedelta(minutes=int(rng.integers(0, 60 * 24 * 10)))
        req = PaymentRequest(
            merchant_id=merchant_id,
            amount=round(amount, 2),
            currency={"US": "USD", "GB": "GBP", "IN": "INR", "DE": "EUR", "BR": "BRL", "AU": "AUD"}[country],
            country=country,
            payment_method=payment_method,
            idempotency_key=str(uuid.uuid4()),
            timestamp=ts,
        )
        requests.append(req)
    return requests


# =====================================================================
# 14. EVALUATION ENGINE
# =====================================================================

@dataclass
class StrategyResult:
    name: str
    n: int
    authorized: int
    successful_value: float
    total_cost: float
    total_latency_ms: float
    total_attempts: int
    gateway_counts: Dict[str, int]

    @property
    def authorization_rate(self) -> float:
        return self.authorized / self.n if self.n else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.n if self.n else 0.0

    @property
    def avg_cost(self) -> float:
        return self.total_cost / self.n if self.n else 0.0

    @property
    def avg_attempts(self) -> float:
        return self.total_attempts / self.n if self.n else 0.0


class EvaluationEngine:
    """Runs the *same* stream of incoming requests through each routing
    strategy (each with its own fresh health monitor / circuit breaker /
    idempotency store, so strategies don't interfere with each other) and
    reports comparative business metrics."""

    def __init__(self, strategies: Dict[str, RoutingStrategy], router_config: RouterConfig):
        self.strategies = strategies
        self.router_config = router_config

    def run(self, requests: List[PaymentRequest]) -> Dict[str, StrategyResult]:
        results: Dict[str, StrategyResult] = {}
        for name, strategy in self.strategies.items():
            health_monitor = GatewayHealthMonitor()
            circuit_breaker = CircuitBreaker()
            idem_store = IdempotencyStore()
            processor = PaymentProcessor(health_monitor)
            router = PaymentRouter(strategy, health_monitor, circuit_breaker, idem_store, processor, self.router_config)

            authorized = 0
            successful_value = 0.0
            total_cost = 0.0
            total_latency = 0.0
            total_attempts = 0
            gateway_counts = {gw: 0 for gw in GATEWAYS}

            for req in requests:
                # Give every strategy its own idempotency key namespace by
                # re-using the same request object (same key) is fine since
                # each strategy has an isolated IdempotencyStore.
                record = router.route(req)
                total_latency += record.total_latency_ms
                total_attempts += len(record.attempts)
                if record.success:
                    authorized += 1
                    successful_value += req.amount
                    total_cost += record.total_cost
                    if record.final_gateway:
                        gateway_counts[record.final_gateway] += 1

            results[name] = StrategyResult(
                name=name, n=len(requests), authorized=authorized, successful_value=successful_value,
                total_cost=total_cost, total_latency_ms=total_latency, total_attempts=total_attempts,
                gateway_counts=gateway_counts,
            )
        return results

    @staticmethod
    def print_comparison(results: Dict[str, StrategyResult]) -> None:
        header = (f"{'Strategy':<14}{'AuthRate':>10}{'SuccessVal':>14}{'AvgCost':>10}"
                  f"{'AvgLatMs':>11}{'AvgAttempts':>13}   {'GatewaySplit'}")
        print(header)
        print("-" * len(header))
        for name, r in results.items():
            split = ", ".join(f"{gw}:{cnt}" for gw, cnt in r.gateway_counts.items())
            print(
                f"{name:<14}{r.authorization_rate*100:>9.2f}%{r.successful_value:>14,.2f}"
                f"{r.avg_cost:>10.3f}{r.avg_latency_ms:>11.1f}{r.avg_attempts:>13.2f}   {split}"
            )


# =====================================================================
# 15. DEMONSTRATIONS: IDEMPOTENCY, CIRCUIT BREAKER / FALLBACK
# =====================================================================

def demonstrate_idempotency(ml_strategy: MLRouter) -> None:
    print("\n--- Idempotency demonstration ---")
    health_monitor = GatewayHealthMonitor()
    circuit_breaker = CircuitBreaker()
    idem_store = IdempotencyStore()
    processor = PaymentProcessor(health_monitor)
    router = PaymentRouter(ml_strategy, health_monitor, circuit_breaker, idem_store, processor, RouterConfig())

    req = PaymentRequest(
        merchant_id="merchant_007", amount=120.50, currency="USD", country="US",
        payment_method="card", idempotency_key="fixed-demo-key-123",
    )

    first = router.route(req)
    second = router.route(req)  # client retries with the SAME idempotency key

    print(f"First submission  -> state={first.state.value:<10} gateway={first.final_gateway} "
          f"from_cache={first.from_idempotency_cache}")
    print(f"Second submission -> state={second.state.value:<10} gateway={second.final_gateway} "
          f"from_cache={second.from_idempotency_cache}")
    print(f"IdempotencyStore contains {len(idem_store)} unique record(s) for 2 submitted requests "
          f"=> no duplicate payment was created.")
    assert first.final_gateway == second.final_gateway
    assert len(idem_store) == 1


def demonstrate_circuit_breaker_and_fallback(ml_strategy: MLRouter) -> None:
    print("\n--- Circuit breaker / health-aware fallback demonstration ---")
    health_monitor = GatewayHealthMonitor()
    circuit_breaker = CircuitBreaker()
    idem_store = IdempotencyStore()
    processor = PaymentProcessor(health_monitor)
    router = PaymentRouter(ml_strategy, health_monitor, circuit_breaker, idem_store, processor, RouterConfig())

    print("Gateway states before outage: " + ", ".join(
        f"{gw}={health_monitor.status_of(gw).value}" for gw in GATEWAYS))

    # Force gateway_a offline and gateway_b degraded, as described in the
    # spec's example (Gateway A: healthy, Gateway B: degraded, Gateway C: offline)
    # -- here we simulate gateway_a going down mid-traffic instead, since it
    # is the ML router's usual top pick, to show fallback in action.
    health_monitor.set_manual_status("gateway_a", GatewayStatus.OFFLINE)
    health_monitor.set_manual_status("gateway_b", GatewayStatus.DEGRADED)
    print("Simulated outage:            " + ", ".join(
        f"{gw}={health_monitor.status_of(gw).value}" for gw in GATEWAYS))

    routed_to = {gw: 0 for gw in GATEWAYS}
    for i in range(25):
        req = PaymentRequest(
            merchant_id="merchant_012", amount=float(50 + i * 7), currency="USD", country="US",
            payment_method="card", idempotency_key=f"outage-demo-{i}",
        )
        record = router.route(req)
        if record.final_gateway:
            routed_to[record.final_gateway] += 1

    print(f"Routing distribution during outage (25 txns): {routed_to}")
    assert routed_to["gateway_a"] == 0, "router should never route to an OFFLINE gateway"
    print("Confirmed: zero transactions were routed to the OFFLINE gateway_a; "
          "traffic correctly failed over to gateway_b/gateway_c.")

    health_monitor.set_manual_status("gateway_a", None)
    health_monitor.set_manual_status("gateway_b", None)


# =====================================================================
# 16. MAIN
# =====================================================================

def main() -> None:
    print("=" * 78)
    print("INTELLIGENT PAYMENT ROUTING & OPTIMIZATION SYSTEM (simulation)")
    print("=" * 78)

    # ---- 1. Historical data generation -------------------------------
    print("\n[1] Generating synthetic historical gateway-performance data...")
    history_df = generate_historical_data(n=20000, seed=RANDOM_SEED)
    print(f"    Generated {len(history_df):,} historical transactions "
          f"across {history_df['merchant_id'].nunique()} merchants, "
          f"{len(COUNTRIES)} countries, {len(GATEWAYS)} gateways.")
    overall_rate = history_df["success"].mean()
    print(f"    Overall historical authorization rate: {overall_rate*100:.2f}%")
    by_gw = history_df.groupby("gateway")["success"].mean().sort_values(ascending=False)
    print("    Historical success rate by gateway:")
    for gw, rate in by_gw.items():
        print(f"      {gw:<12} {rate*100:6.2f}%")

    # ---- 2. Train & compare ML models ---------------------------------
    print("\n[2] Training routing models: Logistic Regression vs Gradient Boosting...")
    model = RoutingModel()
    metrics = model.train(history_df)
    print(f"    {'Model':<22}{'ROC-AUC':>10}{'PR-AUC':>10}{'AuthRate@0.5':>14}{'CalibErr':>11}")
    for name, m in metrics.items():
        marker = "  <- selected" if name == model.best_model_name else ""
        print(f"    {name:<22}{m.roc_auc:>10.4f}{m.pr_auc:>10.4f}{m.auth_rate_at_threshold*100:>13.2f}%{m.calibration_error:>11.4f}{marker}")
    print(f"    Best model by ROC-AUC: {model.best_model_name}")

    # ---- 3. Baselines vs ML routing (Evaluation Engine) ----------------
    print("\n[3] Comparing routing strategies (Fixed vs Rule-based vs ML) on live traffic...")
    live_requests = generate_live_requests(n=4000, merchants=sorted(history_df["merchant_id"].unique()), seed=RANDOM_SEED + 1)

    weights = RoutingWeights(success_weight=1.0, cost_weight=1.0, latency_weight=0.05)
    strategies = {
        "fixed": FixedGatewayRouter(fixed_gateway="gateway_a"),
        "rule_based": RuleBasedRouter(history_df),
        "ml_routing": MLRouter(model, weights),
    }
    router_config = RouterConfig(max_retries=2, backoff_base_ms=20.0, backoff_cap_ms=400.0, simulate_sleep=False)
    engine = EvaluationEngine(strategies, router_config)
    results = engine.run(live_requests)
    EvaluationEngine.print_comparison(results)

    fixed_auth = results["fixed"].authorization_rate
    ml_auth = results["ml_routing"].authorization_rate
    rule_auth = results["rule_based"].authorization_rate
    print("\n    Adding any form of routing (rule-based or ML) over a single fixed gateway with no")
    print("    fallback recovers a large share of otherwise-lost authorizations on this traffic:")
    print(f"      fixed (no fallback)         -> {fixed_auth*100:6.2f}% authorized")
    print(f"      rule-based (hist. lookup)   -> {rule_auth*100:6.2f}% authorized  ({(rule_auth-fixed_auth)*100:+.2f}pp vs fixed)")
    print(f"      ML routing (learned model)  -> {ml_auth*100:6.2f}% authorized  ({(ml_auth-fixed_auth)*100:+.2f}pp vs fixed)")
    print("    Note: rule-based and ML routing land close to each other here because both are")
    print("    effectively approximating the same thing -- P(success | country, payment method,")
    print("    gateway) -- and that signal is strong and low-dimensional enough for a simple")
    print("    historical lookup table to capture most of it. The ML model's edge would typically")
    print("    grow with feature complexity (fraud signals, device/network data, richer segments)")
    print("    and because it, unlike the lookup table, jointly optimizes for cost and latency too")
    print("    via the expected-utility objective, not authorization probability alone.")

    # ---- 4. Idempotency & resilience demonstrations ---------------------
    ml_strategy_for_demo = MLRouter(model, weights)
    demonstrate_idempotency(ml_strategy_for_demo)
    demonstrate_circuit_breaker_and_fallback(ml_strategy_for_demo)

    # ---- 5. Sample end-to-end transaction trace -------------------------
    print("\n--- Sample end-to-end transaction trace ---")
    health_monitor = GatewayHealthMonitor()
    circuit_breaker = CircuitBreaker()
    idem_store = IdempotencyStore()
    processor = PaymentProcessor(health_monitor)
    sample_router = PaymentRouter(MLRouter(model, weights), health_monitor, circuit_breaker, idem_store, processor, router_config)
    sample_req = PaymentRequest(
        merchant_id="merchant_003", amount=249.99, currency="USD", country="US",
        payment_method="card", idempotency_key="trace-demo-1",
    )
    trace = sample_router.route(sample_req)
    print(f"    transaction_id: {trace.request.transaction_id}")
    print(f"    final state:    {trace.state.value}")
    print(f"    final gateway:  {trace.final_gateway}")
    print("    attempts:")
    for i, a in enumerate(trace.attempts, start=1):
        print(f"      attempt {i}: gateway={a.gateway:<10} success={a.success} "
              f"latency_ms={a.latency_ms:.1f} cost={a.cost:.3f} reason={a.reason}")
    print(f"    total_latency_ms: {trace.total_latency_ms:.1f}   total_cost: {trace.total_cost:.3f}")

    print("\n" + "=" * 78)
    print("Simulation complete. See README.md for architecture, methodology, and")
    print("an explicit list of what is simulated vs. not implemented.")
    print("=" * 78)


if __name__ == "__main__":
    main()