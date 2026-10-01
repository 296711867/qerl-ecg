# v3 robustness, oracle bounds and calibration

All runs use the frozen v2b_ce_sqrt classifier recipe, validation-only selection of threshold parameters by mean reward, and the causal Double-Dueling DQN with the same hyperparameters (DQN seed 42). Nothing was retuned after seeing these results.

## Per-run fold means (macro-F1 with abstention as miss; seconds observed)

| Run | Fixed 10 s F1 | Threshold F1 / s | Time-threshold F1 / s | DQN F1 / s | DQN coverage |
|---|---:|---:|---:|---:|---:|
| Frozen split, classifier seed A (main) | 0.5313±0.0351 | 0.5108 / 4.28 | 0.5293 / 3.52 | 0.5346 / 4.03 | 1.000 |
| Frozen split, classifier seed B | 0.5433±0.0281 | 0.5268 / 5.09 | 0.5379 / 3.67 | 0.5534 / 4.34 | 1.000 |
| Frozen split, classifier seed C | 0.5425±0.0231 | 0.5246 / 4.61 | 0.5404 / 3.79 | 0.5490 / 4.24 | 1.000 |
| Re-drawn partition 1 | 0.5561±0.0070 | 0.5439 / 5.36 | 0.5465 / 3.46 | 0.5617 / 4.31 | 1.000 |
| Re-drawn partition 2 | 0.5373±0.0470 | 0.5108 / 4.27 | 0.5310 / 3.35 | 0.5391 / 4.17 | 1.000 |

## Paired patient-cluster bootstrap (pooled out-of-fold predictions)

| Run | DQN − time threshold: ΔF1 | Δs | Time threshold − Fixed 10 s: ΔF1 | Δs |
|---|---:|---:|---:|---:|
| Frozen split, classifier seed A (main) | +0.0078 [-0.0046, +0.0190] | +0.524 [+0.392, +0.654] | +0.0010 [-0.0158, +0.0165] | -4.645 [-4.850, -4.444] |
| Frozen split, classifier seed B | +0.0180 [+0.0062, +0.0303] | +0.622 [+0.514, +0.732] | -0.0065 [-0.0212, +0.0068] | -4.457 [-4.656, -4.256] |
| Frozen split, classifier seed C | +0.0030 [-0.0091, +0.0174] | +0.442 [+0.349, +0.529] | +0.0042 [-0.0122, +0.0185] | -4.334 [-4.525, -4.144] |
| Re-drawn partition 1 | +0.0198 [+0.0079, +0.0321] | +0.855 [+0.762, +0.953] | -0.0145 [-0.0283, -0.0016] | -4.684 [-4.887, -4.495] |
| Re-drawn partition 2 | +0.0054 [-0.0105, +0.0272] | +0.828 [+0.704, +0.951] | -0.0043 [-0.0241, +0.0104] | -4.795 [-4.983, -4.606] |
| Average over 3 classifier seeds | +0.0096 [+0.0022, +0.0172] | +0.530 [+0.451, +0.606] | -0.0004 [-0.0106, +0.0096] | -4.479 [-4.655, -4.298] |
| Average over 3 patient partitions | +0.0110 [+0.0021, +0.0205] | +0.736 [+0.652, +0.817] | -0.0060 [-0.0181, +0.0042] | -4.708 [-4.882, -4.530] |

## DQN versus a time-matched threshold

The matched time-varying rule maximizes validation macro-F1 subject to a validation mean time no longer than the same fold's DQN on validation patients (post hoc, validation-only selection).

| Run | Matched threshold F1 / s | DQN − matched: ΔF1 | Δs |
|---|---:|---:|---:|
| Frozen split, classifier seed A (main) | 0.5252 / 3.38 | +0.0101 [-0.0020, +0.0224] | +0.665 [+0.566, +0.771] |
| Frozen split, classifier seed B | 0.5455 / 4.00 | +0.0121 [-0.0004, +0.0233] | +0.305 [+0.214, +0.403] |
| Frozen split, classifier seed C | 0.5500 / 3.93 | -0.0017 [-0.0126, +0.0085] | +0.291 [+0.206, +0.377] |
| Re-drawn partition 1 | 0.5527 / 3.89 | +0.0141 [+0.0037, +0.0251] | +0.433 [+0.342, +0.535] |
| Re-drawn partition 2 | 0.5393 / 3.87 | -0.0022 [-0.0112, +0.0085] | +0.299 [+0.211, +0.382] |
| Average over 3 classifier seeds | | +0.0068 [-0.0004, +0.0139] | +0.421 [+0.347, +0.497] |
| Average over 3 patient partitions | | +0.0073 [+0.0003, +0.0141] | +0.466 [+0.402, +0.536] |

Averaged rows resample patients once per replicate and average the paired differences across the three runs; runs share patients, so they are not independent.

## Hindsight oracle stopping bounds (main run; uses test labels, descriptive only)

| Quantity | Fold mean ± SD |
|---|---:|
| Oracle-any macro-F1 (stop at first correct prefix) | 0.6724 ± 0.0604 |
| Oracle-any seconds | 2.5684 ± 0.2715 |
| Episodes ever correct at some prefix | 0.8493 ± 0.0352 |
| Oracle-stable macro-F1 (= Fixed 10 s decisions) | 0.5313 ± 0.0351 |
| Oracle-stable seconds (earliest prefix after which the prediction never changes) | 2.8629 ± 0.2868 |
| Fixed 10 s accuracy | 0.7332 ± 0.0339 |
| Fixed 10 s seconds | 8.1543 ± 0.2487 |

## Descriptive operating-point sweep on pooled test trajectories (main run)

Every rule setting is applied to all five held-out folds; this uses test data to draw the frontier and is not a model-selection procedure.

Selected DQN point: 0.5561 at 4.02 s. Selected time-threshold point: 0.5484 at 3.50 s. Rule settings that match or beat the DQN on both axes: 0.

## Reliability (pooled test prefixes, event-weighted, 10 bins)

ECE raw 0.0207; temperature-scaled 0.0158; fold temperatures 1.069, 1.135, 1.050, 1.119, 0.877.
