# Intelligent Payment Routing & Optimization System

A single-file Python simulation of an ML-driven, fault-tolerant payment
routing engine — the kind of system a payments platform (Stripe, Adyen,
Braintree, a card network's acquiring layer, etc.) uses to decide which
downstream gateway/processor should handle each transaction.

Built as a portfolio project for Fintech/SDE and Finance-ML roles
(Stripe, Visa, Mastercard, American Express and similar).

> **Everything here is a simulation.** No real payment processing, card
> data, or money movement occurs anywhere in this program. See
> [Simulated vs. Not Implemented](#simulated-vs-not-implemented) below.

---

## Table of contents

1. [Problem statement](#1-problem-statement)
2. [Payment-routing motivation](#2-payment-routing-motivation)
3. [System architecture](#3-system-architecture)
4. [Data generation](#4-data-generation)
5. [ML model](#5-ml-model)
6. [Routing objective](#6-routing-objective)
7. [Baseline strategies](#7-baseline-strategies)
8. [Reliability mechanisms](#8-reliability-mechanisms)
9. [Idempotency](#9-idempotency)
10. [Circuit breaker](#10-circuit-breaker)
11. [Retry / fallback](#11-retry--fallback)
12. [Evaluation](#12-evaluation)
13. [Installation](#13-installation)
14. [Usage](#14-usage)
15. [Example transaction flow](#15-example-transaction-flow)
16. [Architecture diagram](#16-architecture-diagram)
17. [Limitations](#17-limitations)
18. [Production architecture](#18-production-architecture)
19. [Interview questions](#19-interview-questions)
20. [Simulated vs. not implemented](#simulated-vs-not-implemented)

---

## 1. Problem statement

A payment platform that supports multiple downstream gateways/processors
must decide, for every incoming transaction, **which gateway to send it
to**. Gateways differ in:

- authorization success rate (varies by geography, payment method, and
  transaction size),
- latency,
- processing fee,
- current availability/health,
- regional acquiring strength.

Naively sending all traffic to a single "default" gateway leaves
authorization rate, revenue, and reliability on the table. This project
builds a **routing layer** that scores each candidate gateway per
transaction and picks the one that maximizes expected business value,
while degrading gracefully when a gateway is slow, unhealthy, or down.

## 2. Payment-routing motivation

At small scale, one processor integration is enough. As a platform
grows, it typically adds more processors for:

- **Redundancy** — if one processor has an outage, traffic can shift to
  another instead of every payment failing.
- **Regional coverage** — some processors authorize better in specific
  countries or with specific local payment methods.
- **Cost optimization** — processing fees differ, and so does the
  effective cost per successful transaction once you factor in decline
  rates.
- **Negotiating leverage / risk diversification** — not depending on a
  single vendor.

Once there is more than one processor, *someone* has to decide the
routing logic. This project implements that decision layer end to end:
data → model → objective → routing → reliability → evaluation.

## 3. System architecture

```
Incoming PaymentRequest
        |
        v
  [Validation]
        |
        v
  [IdempotencyStore] -----> cached record? --> return immediately
        |
        v
  [GatewayHealthMonitor] --> filter OFFLINE gateways
        |
        v
  [CircuitBreaker] --------> filter OPEN-circuit gateways
        |
        v
  [RoutingStrategy.rank_candidates]   (Fixed | RuleBased | ML)
        |
        v
  [PaymentProcessor.attempt] --(fail)--> retry with exponential backoff
        |                                  |
      (success)                      pick next candidate gateway
        |                                  |
        v                                  v
  [TransactionRecord: COMPLETED]    [TransactionRecord: FAILED]
        |
        v
  [IdempotencyStore.put] + [GatewayHealthMonitor.record] + [CircuitBreaker.record_result]
```

### Classes (all in `payment_router.py`)

| Class | Responsibility |
|---|---|
| `PaymentRequest` | Incoming transaction request + validation |
| `TransactionRecord` | Mutable record of a transaction's lifecycle, enforces the state machine |
| `Gateway` (modeled via module-level profiles + `PaymentProcessor`) | Simulated downstream processor with its own performance profile |
| `GatewayHealthMonitor` | Rolling per-gateway success rate / latency, derives HEALTHY / DEGRADED / OFFLINE |
| `CircuitBreaker` | Per-gateway CLOSED → OPEN → HALF_OPEN protection with logical-clock recovery |
| `IdempotencyStore` | Prevents duplicate payment records for a retried client request |
| `RoutingModel` | Trains/compares ML classifiers estimating P(success \| transaction, gateway) |
| `RoutingStrategy` (`FixedGatewayRouter`, `RuleBasedRouter`, `MLRouter`) | Pluggable candidate-ranking strategies |
| `PaymentProcessor` | Simulates actually calling a gateway (latency, timeout, outcome sampling) |
| `PaymentRouter` | Orchestrates validation → idempotency → health/circuit filtering → scoring → attempt/retry/fallback |
| `EvaluationEngine` | Runs the same traffic through every strategy and reports comparative metrics |

## 4. Data generation

`generate_historical_data()` produces synthetic historical transactions
(20,000 by default) with a **hidden ground-truth performance surface**
(`true_success_probability`, `true_latency_ms`, `true_processing_cost`)
that varies by:

- **geography** — per-gateway, per-country modifiers (e.g. one gateway's
  acquiring is stronger in the US/EU than in Brazil),
- **payment method** — card / wallet / bank_transfer / bnpl each have
  different per-gateway modifiers,
- **transaction amount** — large amounts see a modest authorization
  penalty (risk engines bite harder),
- **time of day** — a small overnight dip (maintenance-window effect),
- **merchant** — merchant-specific amount distributions,
- **gateway load** — a simulated load multiplier that degrades
  authorization slightly under heavier traffic.

Both the historical-data generator and the real-time simulator sample
from this same surface, so the ML model is learning a genuine (if noisy)
relationship rather than fitting noise.

Each row contains: `transaction_id, merchant_id, amount, country,
currency, payment_method, gateway, timestamp, success, latency_ms,
processing_cost` (plus derived `hour`, `day_of_week`).

## 5. ML model

`RoutingModel` trains and compares two classifiers estimating
**P(payment_success | transaction, gateway)**:

1. Logistic Regression
2. Gradient Boosting (`sklearn.ensemble.GradientBoostingClassifier`)

Both use a `ColumnTransformer` (one-hot encoding for `gateway`,
`country`, `payment_method`; passthrough for `amount`, `hour`,
`day_of_week`) inside an `sklearn.Pipeline`, trained on a 75/25
stratified train/test split.

**Actual results from `python payment_router.py`** (seed = 42, 20,000
historical rows):

| Model | ROC-AUC | PR-AUC | Auth rate @ 0.5 threshold | Calibration error |
|---|---|---|---|---|
| Logistic Regression | 0.5742 | 0.8012 | 75.48% | 0.0222 |
| **Gradient Boosting (selected)** | **0.5822** | **0.8055** | 75.46% | **0.0132** |

Gradient Boosting is selected (higher ROC-AUC, better-calibrated
probabilities) and used by `MLRouter`. Note the modest ROC-AUC: this
reflects that gateway choice is a real but secondary driver of success
in the synthetic ground truth relative to country/payment-method/amount,
which is intentional — an unrealistically separable problem would be a
weaker demonstration of the routing objective downstream.

## 6. Routing objective

Every weight in the objective is explicit and configurable via
`RoutingWeights` — nothing is a buried, unexplained constant in the
routing logic:

```python
utility = success_weight * P(success) * amount
          - cost_weight    * processing_cost
          - latency_weight * (latency_ms / 1000)
```

Defaults (`success_weight=1.0, cost_weight=1.0, latency_weight=0.05`)
reflect one reasonable business stance — expected captured revenue
dominates, cost is weighted equally against it in absolute dollar terms,
and latency carries a small penalty per second — but any business can
retune these for their own priorities (e.g. a business highly sensitive
to checkout latency would raise `latency_weight`).

`MLRouter` computes this utility for every health/circuit-eligible
candidate gateway and ranks best-first.

## 7. Baseline strategies

| Strategy | Behavior |
|---|---|
| **Fixed** | Routes 100% of traffic to a single hard-coded gateway, **with no fallback** — representative of a platform with one processor integration wired in directly. |
| **Rule-based** | Looks up the historical observed success rate for the transaction's `(country, payment_method)` segment per gateway, breaks ties by lower cost. No ML, just aggregation. |
| **ML routing** | Ranks every healthy candidate by expected utility from the trained `RoutingModel`. |

**Actual results** (4,000 simulated live transactions, seed = 43):

| Strategy | Auth rate | Successful value | Avg cost | Avg latency (ms) | Avg attempts | Gateway split |
|---|---|---|---|---|---|---|
| fixed | 76.02% | $120,798.38 | 1.104 | 277.5 | 0.96 | a:3041, b:0, c:0 |
| rule_based | 98.70% | $157,505.23 | 1.331 | 404.8 | 1.23 | a:2015, b:1167, c:766 |
| ml_routing | 98.40% | $157,068.42 | 1.315 | 427.9 | 1.24 | a:2358, b:1320, c:258 |

**Takeaways from this run:**

- Simply *having* fallback (rule-based or ML) over a single fixed
  gateway with none recovers **~22 percentage points** of authorization
  rate and roughly **$36k** more successfully captured transaction value
  on this traffic sample — the single biggest lever here is having
  *any* routing/fallback layer at all.
- Rule-based and ML routing land close to each other. Both are
  effectively approximating the same thing — P(success | country,
  payment method, gateway) — and that signal is strong and
  low-dimensional enough for a simple historical lookup table to
  capture most of it on this synthetic dataset. In a real system, the
  ML model's edge would typically grow with feature complexity (fraud
  signals, device/network data, richer segments) and because it, unlike
  the lookup table, jointly optimizes for cost and latency too via the
  expected-utility objective rather than authorization probability
  alone.
- `avg_attempts` shows fixed routing gives up after 0.96 attempts on
  average (it has nowhere to fall back to), while routing-enabled
  strategies use ~1.23–1.24 attempts on average to capture that extra
  authorization rate.

*(Exact numbers will vary slightly if you change `RANDOM_SEED` or the
traffic volume, but the qualitative pattern — fixed routing leaves
authorization rate on the table, rule-based and ML routing recover
most of it — is stable across seeds in this simulation.)*

## 8. Reliability mechanisms

All implemented directly in `payment_router.py`:

- **Idempotency keys** — `IdempotencyStore`
- **Request timeouts** — `PaymentProcessor(timeout_ms=...)`
- **Retries** — `RouterConfig.max_retries`
- **Exponential backoff** — `exponential_backoff_delay()` (AWS-style
  full jitter)
- **Circuit breaker** — `CircuitBreaker` (CLOSED → OPEN → HALF_OPEN)
- **Gateway health tracking** — `GatewayHealthMonitor`
- **Fallback routing** — `PaymentRouter.route()`'s retry loop pops the
  next-ranked candidate gateway on failure

## 9. Idempotency

The same client-supplied `idempotency_key` submitted twice must not
create two successful payment records — mirroring how real payment APIs
(e.g. Stripe's `Idempotency-Key` header) prevent duplicate charges when
a client retries after a network error.

**Actual output:**

```
--- Idempotency demonstration ---
First submission  -> state=COMPLETED  gateway=gateway_a from_cache=True
Second submission -> state=COMPLETED  gateway=gateway_a from_cache=True
IdempotencyStore contains 1 unique record(s) for 2 submitted requests => no duplicate payment was created.
```

*(Both attempts show `from_cache=True` because in this implementation
even the first response is a reference to the single stored record —
what matters, and what's asserted in code, is that both calls resolve
to the exact same `TransactionRecord` and the store holds exactly one
entry for two submissions.)*

## 10. Circuit breaker

Classic per-gateway `CLOSED → OPEN → HALF_OPEN` state machine:
`failure_threshold` consecutive failures opens the circuit; after
`recovery_timeout` the circuit moves to `HALF_OPEN` and allows one
trial request through; success closes it again, failure re-opens it.

**Design note (a bug found and fixed during development):** the
recovery timeout is measured on a **logical clock** — a counter
incremented once per routed request by `PaymentRouter`, not
`time.monotonic()`. An earlier version used wall-clock time, but since
thousands of simulated transactions are routed in milliseconds of real
time, a wall-clock cooldown would never elapse within a single
evaluation run, meaning any gateway that ever tripped the breaker would
stay locked out for the rest of the simulation. A request-count-based
clock keeps recovery behavior meaningful regardless of how fast the
simulation executes; in a real production deployment you would use
wall-clock time instead, since real request arrival isn't
instantaneous.

A related bug was found and fixed in `GatewayHealthMonitor`: an earlier
version *automatically* inferred an `OFFLINE` status from a rolling
success rate dropping below a threshold — but an `OFFLINE` gateway
receives no traffic, so it could never collect the data needed to
recover, creating a permanent lockout from ordinary statistical noise.
The fix: automatic health scoring is limited to `HEALTHY`/`DEGRADED`;
`OFFLINE` is only ever reached via an explicit simulated outage (see
below) or algorithmically via the `CircuitBreaker`, which has a real
recovery path.

## 11. Retry / fallback

**Actual output — simulated outage demonstration:**

```
--- Circuit breaker / health-aware fallback demonstration ---
Gateway states before outage: gateway_a=HEALTHY, gateway_b=HEALTHY, gateway_c=HEALTHY
Simulated outage:            gateway_a=OFFLINE, gateway_b=DEGRADED, gateway_c=HEALTHY
Routing distribution during outage (25 txns): {'gateway_a': 0, 'gateway_b': 6, 'gateway_c': 17}
Confirmed: zero transactions were routed to the OFFLINE gateway_a; traffic correctly failed over to gateway_b/gateway_c.
```

`gateway_a` is forced `OFFLINE` and `gateway_b` `DEGRADED` (matching the
example in the spec: "Gateway A: healthy, Gateway B: degraded, Gateway
C: offline" — here simulating gateway_a going down instead, since it is
the router's usual top pick, to actually exercise the fallback path).
The router correctly routes **zero** transactions to the offline
gateway and shifts traffic to the remaining healthy/degraded gateways.

## 12. Evaluation

Run `python payment_router.py` for the full evaluation. It:

1. Generates historical data and reports authorization rate by gateway.
2. Trains and compares both ML models (ROC-AUC, PR-AUC, calibration).
3. Runs `EvaluationEngine` across Fixed / Rule-based / ML routing on
   identical simulated live traffic, with **independent** health
   monitors, circuit breakers, and idempotency stores per strategy so
   the strategies don't interfere with each other.
4. Demonstrates idempotency.
5. Demonstrates circuit-breaker-driven fallback during a simulated
   outage.
6. Prints a sample end-to-end transaction trace.

The full model-training and strategy-comparison metrics are in
sections 5 and 7 above, taken directly from a real run of this file.

## 13. Installation

Requires Python 3.9+ and:

```bash
pip install numpy pandas scikit-learn
```

No other dependencies. No notebooks, no frontend, no additional Python
modules — everything lives in `payment_router.py` as specified.

## 14. Usage

```bash
python payment_router.py
```

Runs the full pipeline (data generation → model training → strategy
comparison → reliability demonstrations → sample trace) and prints a
report to stdout. Takes well under a minute on a laptop CPU.

To use pieces of it programmatically:

```python
from payment_router import (
    RoutingModel, MLRouter, RoutingWeights, PaymentRouter, PaymentRequest,
    GatewayHealthMonitor, CircuitBreaker, IdempotencyStore, PaymentProcessor,
    RouterConfig, generate_historical_data,
)

history = generate_historical_data(n=20000)
model = RoutingModel()
model.train(history)

strategy = MLRouter(model, RoutingWeights())
router = PaymentRouter(
    strategy, GatewayHealthMonitor(), CircuitBreaker(),
    IdempotencyStore(), PaymentProcessor(GatewayHealthMonitor()), RouterConfig(),
)

request = PaymentRequest(
    merchant_id="merchant_001", amount=150.00, currency="USD",
    country="US", payment_method="card", idempotency_key="order-12345",
)
record = router.route(request)
print(record.state, record.final_gateway, record.total_cost)
```

## 15. Example transaction flow

**Actual output — a real trace from a run of this file:**

```
--- Sample end-to-end transaction trace ---
    transaction_id: 60c27994-9838-4095-bd8e-f8702beee8db
    final state:    COMPLETED
    final gateway:  gateway_a
    attempts:
      attempt 1: gateway=gateway_a  success=True latency_ms=296.0 cost=7.550 reason=authorized
    total_latency_ms: 296.0   total_cost: 7.550
```

This transaction succeeded on the first attempt. A declined/failed
first attempt would show a second `attempts` entry for the fallback
gateway, and `total_latency_ms` would include the exponential-backoff
delay between attempts (see `PaymentRouter.route()`).

## 16. Architecture diagram

See [section 3](#3-system-architecture) above for the full request-flow
diagram.

## 17. Limitations

This is a **simulation for demonstration purposes**. Specific
limitations:

- The "ground truth" performance surface (`true_success_probability`
  and friends) is a synthetic authoring choice, not derived from real
  payment industry data. The qualitative patterns (routing beats no
  routing, ML and rule-based can converge when the underlying signal is
  simple) are illustrative, not empirical claims about real gateways.
- The ML model's ROC-AUC (~0.58) is modest by design — gateway choice
  is a secondary driver of success relative to country/method/amount in
  the synthetic data, which is realistic (gateway is rarely the
  dominant factor in real authorization decisions either) but means the
  model should not be read as "highly predictive."
- No real network calls, no real gateway SDKs/APIs, no real card data
  anywhere.
- The circuit breaker's recovery timeout uses a request-count logical
  clock for simulation speed; a production system would use wall-clock
  time (and likely a sliding time window rather than a simple counter).
- No concurrency/threading — the simulation is single-threaded and
  synchronous; a production router would need to handle concurrent
  requests safely (thread-safe or async-safe health/circuit-breaker
  state).
- No persistence — all state (idempotency store, health monitor,
  circuit breaker) lives in memory and resets on restart. A production
  system needs a durable idempotency store (e.g. a database with a
  unique constraint on the idempotency key) and shared, cross-instance
  gateway health state.
- No authentication, encryption, PCI-DSS scope, or any of the
  compliance machinery a real payments system requires.

## 18. Production architecture

A real version of this system would differ in several important ways:

- **Idempotency store** → a database (e.g. Postgres/DynamoDB) with a
  unique constraint on `idempotency_key`, not an in-memory dict, so it
  survives restarts and is consistent across multiple router instances.
- **Gateway health state** → shared across all router instances (e.g.
  Redis), updated via a fast, low-latency store, since health decisions
  need to be consistent cluster-wide, not per-process.
- **Model serving** → the routing model would be served from a
  dedicated low-latency inference service (or compiled to something
  like ONNX) rather than an in-process scikit-learn pipeline, and
  retrained on a schedule against fresh production outcome data with a
  proper feature store.
- **Actual gateway integration** → real SDKs/APIs per processor (with
  their own auth, request signing, and response parsing), behind an
  adapter interface so `PaymentRouter` doesn't need to know gateway-
  specific details.
- **Observability** → structured logging, distributed tracing, and
  dashboards on authorization rate, latency, and circuit-breaker state
  per gateway, with alerting on anomalies.
- **Compliance** → PCI-DSS scope reduction (e.g. tokenization, not
  touching raw card data in the routing layer at all), audit logging,
  and regulatory reporting.
- **A/B testing / gradual rollout** — new routing strategies would be
  shadow-tested or rolled out to a small traffic percentage before
  full deployment, with statistical significance testing on the
  authorization-rate delta.

## 19. Interview questions

Questions this project is designed to let you speak to (and that an
interviewer might reasonably ask a candidate who built it):

1. Why is authorization rate alone an insufficient metric for
   evaluating a routing strategy? *(See section 7 — with retries
   enabled, authorization rate can converge across strategies; cost,
   latency, and attempts-per-transaction differentiate them.)*
2. How would you prevent the circuit breaker or health monitor from
   creating a self-reinforcing lockout on a gateway? *(See section 10 —
   two such bugs were found and fixed during development of this very
   project.)*
3. Why use idempotency keys instead of deduplicating on transaction ID
   alone? *(Because the client generates the idempotency key before
   knowing whether the first attempt succeeded, and can safely retry
   with the same key without knowing the outcome of the first attempt.)*
4. How would you calibrate the routing objective's weights
   (`success_weight`, `cost_weight`, `latency_weight`) in a real
   business? *(Tie them to actual unit economics — e.g. cost_weight
   should reflect real margin impact per dollar of processing fee, and
   latency_weight should be calibrated against measured checkout
   abandonment rate per additional second of latency.)*
5. How would you detect model drift in production, and what would you
   do about it? *(Monitor live calibration curves against fresh
   outcome data; retrain on a schedule; alert if authorization rate for
   the top-ranked gateway diverges from its predicted probability by
   more than some threshold.)*
6. Why train two models instead of just picking one? *(To have an
   empirical, not assumed, basis for the production model choice — see
   the ROC-AUC/PR-AUC/calibration comparison in section 5 — and because
   logistic regression's coefficients are more interpretable, which
   matters for explaining declines and for regulatory/model-risk
   review even if gradient boosting wins on raw performance.)*
7. What would break first if you 100x'd transaction volume, and how
   would you fix it? *(The in-memory idempotency store and health
   monitor state are the first bottlenecks/single points of failure —
   see section 18's production architecture notes.)*

---

## Simulated vs. not implemented

**SIMULATED:**
- Payment gateways (`gateway_a/b/c`) and their performance profiles
- Transactions and their outcomes
- Gateway latency, load, and health status
- Authorization success/failure sampling
- Circuit breaker and health-monitor-driven failover
- An outage scenario (forced OFFLINE/DEGRADED gateway states)

**NOT IMPLEMENTED:**
- Actual payment processing or money movement
- PCI-DSS compliance or any handling of real card data
- Real card network or processor integrations/SDKs
- Real financial settlement, reconciliation, or reporting
- Authentication, encryption, or any production security controls

This project does not claim production readiness. It is a
demonstration of ML-based routing methodology and reliable backend
system design (idempotency, retries, circuit breakers, health-aware
failover) in a controlled, fully synthetic environment.