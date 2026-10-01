# Exploratory validation-selected observation-budget analysis

Each fold selects a time-varying threshold on validation patients by maximum nine-class macro-F1 subject to a validation mean-time budget. Each selected configuration is evaluated once on its held-out fold; the same test folds are reused across the seven budgets. These are post hoc exploratory analyses of the frozen v2 trajectories, not a new confirmatory test set.

| Validation budget (s) | Feasible folds | Test macro-F1, fold mean ± SD | Test observed seconds, fold mean ± SD |
|---:|---:|---:|---:|
| 2 | 5 | 0.4630 ± 0.0228 | 1.932 ± 0.095 |
| 3 | 5 | 0.5062 ± 0.0401 | 2.721 ± 0.136 |
| 4 | 5 | 0.5197 ± 0.0367 | 3.271 ± 0.616 |
| 5 | 5 | 0.5256 ± 0.0388 | 3.809 ± 0.697 |
| 6 | 5 | 0.5287 ± 0.0360 | 4.608 ± 1.328 |
| 7 | 5 | 0.5288 ± 0.0358 | 4.854 ± 1.650 |
| 8 | 5 | 0.5283 ± 0.0366 | 5.003 ± 1.897 |

## Accuracy by maximum available annotated duration

Accuracy is pooled across the five held-out folds within each duration stratum. The stopping rule here is the original reward-selected time threshold, not a point selected from the budget curve.

| Available duration | Episodes | Fixed 10 s accuracy | Time-threshold accuracy | Time-threshold mean observed seconds |
|---|---:|---:|---:|---:|
| 1–2 s | 444 | 0.770 | 0.773 | 1.306 |
| 3–4 s | 412 | 0.786 | 0.791 | 2.711 |
| 5–9 s | 728 | 0.695 | 0.718 | 3.582 |
| 10+ s | 3209 | 0.730 | 0.735 | 3.882 |

## Pooled class outcomes for the original reward-selected rule

Each event has one out-of-fold prediction. Class F1 is one-versus-rest on pooled held-out patients, so it is not the unweighted mean of fold F1 values.

| Rhythm | Episodes | Patients | Fixed 10 s F1 | Time-threshold F1 | Time-threshold recall |
|---|---:|---:|---:|---:|---:|
| N | 2650 | 370 | 0.823 | 0.832 | 0.847 |
| AFIB/AFL | 320 | 111 | 0.698 | 0.650 | 0.728 |
| AVB | 32 | 10 | 0.271 | 0.308 | 0.375 |
| SND | 230 | 66 | 0.411 | 0.415 | 0.361 |
| SR-mPAC-BT | 546 | 85 | 0.482 | 0.527 | 0.469 |
| SR-mPVC-BT | 611 | 109 | 0.715 | 0.719 | 0.722 |
| SVTA | 187 | 96 | 0.660 | 0.655 | 0.615 |
| VT | 179 | 74 | 0.866 | 0.829 | 0.922 |
| MAT | 38 | 23 | 0.000 | 0.000 | 0.000 |
