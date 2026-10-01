# Random-horizon DQN: validation-only pilot

Training events were independently truncated at a uniformly sampled integer horizon from 1 to their original annotated duration on each of the 12 training passes. Architecture, reward, classifier trajectories and validation checkpoint rule matched the original causal DQN. No test trajectories were evaluated for this pilot. The same validation patients selected each checkpoint and are reported here, so these comparisons are optimistic and exploratory.

| Policy | Validation macro-F1 (5-fold mean±SD) | Observation seconds | Reward | Forced at annotated end |
|---|---:|---:|---:|---:|
| Original | 0.5637±0.0500 | 4.0123±0.1150 | 0.1939±0.1154 | 847/4793 |
| Random horizon | 0.5694±0.0521 | 4.2755±0.7233 | 0.1878±0.1275 | 776/4793 |

Pooled validation patient-bootstrap difference, random horizon minus original: macro-F1 +0.0057 (95% CI [-0.0010, +0.0140]); time +0.242 s (95% CI [+0.141, +0.330]). These intervals do not account for validation checkpoint selection and are not confirmation.

## Fold detail

| Fold | N | Original F1 / s / forced | Random horizon F1 / s / forced |
|---:|---:|---|---|
| 0 | 1004 | 0.6261 / 3.992 / 234 | 0.6413 / 3.119 / 138 |
| 1 | 916 | 0.5672 / 3.916 / 149 | 0.5603 / 4.177 / 147 |
| 2 | 819 | 0.5339 / 4.182 / 123 | 0.5348 / 5.070 / 142 |
| 3 | 1009 | 0.5934 / 4.066 / 139 | 0.6001 / 4.565 / 150 |
| 4 | 1045 | 0.4980 / 3.905 / 202 | 0.5103 / 4.448 / 199 |

Go/no-go: improvement must remain after comparison with a matched simple threshold, across classifier seeds and on new patients or a genuinely independent dataset; otherwise retain the benchmark interpretation. Random truncation alone is not yet a novel algorithm.
