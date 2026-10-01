Supplementary Code 1 for the v5 working manuscript

Repository: https://github.com/296711867/qerl-ecg
Submission snapshot: v5-submission-2026-10-01-r1
No archive DOI is claimed. No software license has been assigned to the original
analysis code; public visibility alone does not grant an open-source license.
Source-data-derived tables retain the source CC BY 4.0 attribution requirements.
The Elsevier class is not included in this analysis repository.

CONTENTS
Analysis source, saved configuration, frozen patient splits, derived episode and
quality tables, v3 summary results, v4 decisions, and editable figure sources.
SHA256SUMS.txt identifies exactly which files were packaged.
The source includes historical helpers; the sequence below identifies the active
onset-anchored causal-DQN analysis. v1 results are superseded and must not be used.

INPUTS AND ENVIRONMENT
Obtain VitalDB Arrhythmia Database v1.0.0 (10.13026/axd6-wm13) annotations/metadata,
and the corresponding lead-II waveforms from VitalDB v1.0.0 (10.13026/czw8-9p62).
Prepare data/annotations and data/raw as specified by configs/project.yaml;
src/data/download_waveforms.py and clean_annotations.py implement ingestion.
The derived segments, badq_intervals and event_episodes_v2 parquet files and frozen
splits are included. Raw waveforms, posterior trajectory files and model checkpoints
are not included. Recreate them by training/inference or obtain their exact study
versions. Training requires scientific Python, PyTorch, NumPy, Pandas, SciPy,
scikit-learn, PyYAML, pyarrow, matplotlib, and vitaldb for downloading waveforms.
The PowerShell runners currently name C:\Asoftware\anaconda3\python.exe; adapt
that executable path to your environment. GPU availability and RNG/library versions
can affect rerun results. No fully pinned original environment is claimed.

ACTIVE SEQUENCE (from the qerl-ecg root)
1. scripts/run_event_v2.ps1 -Variant v2b_ce_sqrt
2. scripts/run_event_causal.ps1
3. scripts/run_v3.ps1
4. python scripts/analyze_v3.py
5. python scripts/v4_collect.py
6. python scripts/make_v4_figures.py fig1 fig3 fig4 fig5 figS1 figS2
7. python scripts/benchmark_latency.py
8. python tests/test_v2_core.py

Figure 2 was maintained separately; use its supplied SVG/PDF. The default figure
command would overwrite it. Plotting writes to paper/v4/figures; v5 preserves those
verified vector outputs. Generation of ECG example panels still needs raw waveforms.
The tests above validate the analysis code; they were not rerun during the v5
formatting iteration. Training was not rerun and numbers retain the v3 provenance.
