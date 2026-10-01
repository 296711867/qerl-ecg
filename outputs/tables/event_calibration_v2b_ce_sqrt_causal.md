# Exploratory prefix-posterior calibration

Each fold fits one scalar temperature on validation prefixes by weighted NLL. Each event contributes total weight one across its available prefixes. The same validation patients select raw and calibrated threshold parameters separately by reward; test patients are only evaluated. This is post hoc and does not provide independent confirmation.

| Outcome | Raw, fold mean ± SD | Calibrated, fold mean ± SD |
|---|---:|---:|
| 10-bin ECE | 0.0436 ± 0.0146 | 0.0408 ± 0.0104 |
| Multiclass Brier | 0.4357 ± 0.0326 | 0.4347 ± 0.0306 |
| Threshold macro-F1 | 0.5293 ± 0.0401 | 0.5239 ± 0.0418 |
| Threshold observed seconds | 3.5205 ± 0.6290 | 3.4432 ± 0.6773 |
| Threshold mean reward | 0.1502 ± 0.1235 | 0.1459 ± 0.1225 |

Validation-fitted temperatures: 1.069, 1.135, 1.050, 1.119, 0.877.
