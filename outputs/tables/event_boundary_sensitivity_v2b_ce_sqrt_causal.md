# Exploratory annotated-boundary sensitivity audit

The model observes only causal prefixes, but the retrospective simulator forces a classification when an annotated rhythm segment ends. Here *forced* means a policy still requested observation when that segment ended before the common 10 s cap. Active stops exactly at the end are not counted as forced. This analysis was designed after seeing the main results; it is descriptive and cannot establish a continuous-monitoring effect.

## Full cohort: forced stops

| Policy | Forced / 4,793 | Forced fraction | Accuracy in forced events |
|---|---:|---:|---:|
| Fixed 10s | 1584 | 33.048% | 0.7399 |
| Time threshold | 462 | 9.639% | 0.6407 |
| RL | 869 | 18.131% | 0.6525 |

Forced counts by class (N, AFIB/AFL, AVB, SND, SR-mPAC-BT, SR-mPVC-BT, SVTA, VT, MAT):

- Fixed 10s: [701, 46, 0, 63, 220, 217, 166, 169, 2]
- Time threshold: [240, 21, 0, 23, 42, 12, 96, 28, 0]
- RL: [337, 31, 0, 50, 109, 77, 155, 109, 1]

## Common-horizon risk sets

Only episodes with at least the stated duration are included. Each is then capped at the same horizon for every policy; the original validation-selected threshold and DQN checkpoints are reused without retuning. This is a subset sensitivity analysis, not a comparison on all events or a new held-out test.

| Horizon | Events | Policy | Five-fold macro-F1 | Five-fold time (s) | Pooled RL − threshold macro-F1 (patient bootstrap 95% CI) | RL − threshold time (s; 95% CI) |
|---:|---:|---|---:|---:|---:|---:|
| 3 s | 4349 | Fixed horizon | 0.4777±0.0533 | 3.0000±0.0000 |  |  |
| 3 s | 4349 | Time threshold | 0.4798±0.0518 | 2.7111±0.1496 |  |  |
| 3 s | 4349 | RL | 0.4710±0.0521 | 2.6487±0.1669 | -0.0100 [-0.0180, -0.0028] | -0.058 [-0.115, -0.003] |
| 5 s | 3937 | Fixed horizon | 0.4622±0.0800 | 5.0000±0.0000 |  |  |
| 5 s | 3937 | Time threshold | 0.4626±0.0823 | 3.4780±0.4323 |  |  |
| 5 s | 3937 | RL | 0.4670±0.0796 | 3.5238±0.2763 | +0.0057 [-0.0112, +0.0200] | +0.052 [-0.045, +0.145] |
| 8 s | 3438 | Fixed horizon | 0.4548±0.0511 | 8.0000±0.0000 |  |  |
| 8 s | 3438 | Time threshold | 0.4497±0.0513 | 3.8545±0.6847 |  |  |
| 8 s | 3438 | RL | 0.4747±0.0644 | 4.2773±0.4183 | +0.0230 [-0.0008, +0.0422] | +0.432 [+0.295, +0.558] |

Selection of long-enough episodes changes the class and duration mix. The original classifier was trained on the full eligible cohort and may still encode duration-correlated patterns. None of these estimates are prospective or independently confirmatory.
