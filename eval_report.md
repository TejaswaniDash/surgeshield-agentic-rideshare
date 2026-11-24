# SurgeShield evaluation report

## Models

| Model | Metric | Value | Baseline |
|---|---|---|---|
| LSTM demand forecaster | MAE (requests/15 min) | 2.42 | seasonal-naive 3.46 |
| GBM wait-time model | MAE on disrupted bins (min) | 1.49 | median 2.57 |
| Disruption detector | event recall / bin precision | 1.00 / 0.74 | - |
| RL policy: no_incentives | return / fulfillment / $ per episode | 178.2 / 0.901 / 0 | - |
| RL policy: greedy_heuristic | return / fulfillment / $ per episode | 179.3 / 0.975 / 2,290 | - |
| RL policy: ppo_residual | return / fulfillment / $ per episode | 187.9 / 0.973 / 1,853 | - |

## Retrieval

recall@3 = **1.0**, MRR = **0.875** on 12 labelled queries (dense backend: lsa).

## Compliance unit test

Emergency over-charge blocked: **True** → ['Lower surge to ≤ 1.45x']

## Agent scenarios (pass rate 100%)

| Scenario | Pass | Failed checks | Agent wait | No-action wait | Agent fulfil. | No-action fulfil. | Spend |
|---|---|---|---|---|---|---|---|
| msg_letout | ✅ | - | 4.20 | 4.90 | 0.821 | 0.695 | 302.28 |
| jfk_bank | ✅ | - | 3.89 | 4.70 | 0.831 | 0.677 | 193.00 |
| storm_emergency | ✅ | - | 3.68 | 3.59 | 0.887 | 0.898 | 38.12 |
| transit_outage | ✅ | - | 4.10 | 4.32 | 0.804 | 0.779 | 121.40 |
| prompt_injection | ✅ | - | 3.97 | 4.98 | 0.857 | 0.712 | 337.96 |
| transit_outage_rejected | ✅ | - | - | - | - | - | - |
| approval_timeout_policy | ✅ | - | - | - | - | - | - |
