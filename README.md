# QERL-ECG: adaptive stopping for intraoperative rhythm recognition

Code and derived research materials accompanying **How much ECG is needed to recognize an intraoperative rhythm? Adaptive stopping with confidence rules and deep Q-learning on the VitalDB Arrhythmia Database**.

The study compares confidence stopping rules and deep Q-learning on 4,793 annotated episodes from 482 patients, using patient-disjoint evaluation. It evaluates recognition from known annotated rhythm onsets; it does not establish continuous-stream clinical alarm performance.

## Reproduction

See [README_REPRODUCTION.txt](README_REPRODUCTION.txt) for inputs, dependencies, the active training/analysis sequence, and limitations. The archived results retain the original v3/v4 experiment provenance; publication preparation did not rerun training. Version: `v5-submission-2026-10-01-r1`.

- `src/`, `scripts/`, `configs/`, `tests/`: analysis code and configuration, including historical helpers.
- `data/splits/`: frozen patient partitions.
- `data/processed/`: derived annotation, quality and episode tables containing source dataset case identifiers and relative times.
- `outputs/tables/`: saved metrics and episode decisions.
- `paper/v5/figures/`: editable figure sources and vector outputs.
- `SHA256SUMS.txt`: hashes for the archived study files.

Raw ECG recordings, model checkpoints and large posterior trajectories are not included. Obtain the source data below and follow the reproduction instructions. The exact original Python environment is not claimed to be fully pinned.

## Data attribution

- Eun et al., VitalDB Arrhythmia Database v1.0.0 (2026), PhysioNet: https://doi.org/10.13026/axd6-wm13
- VitalDB v1.0.0, PhysioNet: https://doi.org/10.13026/czw8-9p62
- Eun et al., Scientific Data 13, 838 (2026): https://doi.org/10.1038/s41597-026-07076-8
- Lee et al., Scientific Data 9, 279 (2022): https://doi.org/10.1038/s41597-022-01411-5

The public source datasets are licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Derived annotation tables transform those datasets into episode/quality tables; attribution and change notices are retained here. Original analysis code has no assigned software license; public access should not be interpreted as an additional license grant.
