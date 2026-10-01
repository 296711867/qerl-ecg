"""Count long-stream rhythm spans in verified PhysioNet LTAFDB annotations."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/external/ltafdb"


def main():
    records = (DATA / "RECORDS").read_text().split()
    expected = {name: digest for digest, name in
                (line.split(maxsplit=1) for line in (DATA / "SHA256SUMS.txt").read_text().splitlines())}
    spans = []
    record_seconds = 0.0
    for record in records:
        for ext in ("atr", "hea"):
            name = f"{record}.{ext}"
            if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != expected[name]:
                raise ValueError(f"PhysioNet SHA256 mismatch: {name}")
        header = (DATA / f"{record}.hea").read_text().splitlines()[0].split()
        fs, n_samples = float(header[2]), int(header[3])
        record_seconds += n_samples / fs
        ann = wfdb.rdann(str(DATA / record), "atr")
        marks = [(int(sample), str(note).strip().strip("\x00"))
                 for sample, note in zip(ann.sample, ann.aux_note)
                 if str(note).strip().startswith("(")]
        for (start, label), (end, _) in zip(marks, marks[1:] + [(n_samples, "END")]):
            if end > start:
                spans.append(dict(record=record, label=label, seconds=(end - start) / fs,
                                  start_sample=start, end_sample=end))
        print(record, len(marks), flush=True)
    by_label = defaultdict(list)
    for span in spans:
        by_label[span["label"]].append(span)
    lines = ["# Long-Term AF Database rhythm annotation feasibility audit", "",
             f"Source: [PhysioNet LTAFDB v1.0.0](https://physionet.org/content/ltafdb/1.0.0/), official anonymous S3. All {2 * len(records)} .atr/.hea files passed the dataset SHA256 manifest. Waveforms were downloaded and verified separately; this annotation audit reads only .atr/.hea.", "",
             "Each rhythm span begins at an `atr` transition marker and ends at the next transition or record end. Pre-first-marker time is excluded. Counts are retrospective annotation transitions, not independent patients or observed online detections.", "",
             "| Rhythm tag | Spans | Records | Total hours | Median span (s) | Spans <1 s |",
             "|---|---:|---:|---:|---:|---:|"]
    for tag in sorted(by_label, key=lambda x: -len(by_label[x])):
        group = by_label[tag]
        d = np.array([s["seconds"] for s in group])
        lines.append(f"| `{tag}` | {len(group)} | {len({s['record'] for s in group})} | "
                     f"{d.sum()/3600:.3f} | {np.median(d):.2f} | {int((d < 1).sum())} |")
    target = [s for s in spans if s["label"] in ("(VT", "(SVTA")]
    labelled_seconds = sum(s["seconds"] for s in spans)
    lines += ["", f"Total waveform time: {record_seconds/3600:.1f} h; transition-labelled time: {labelled_seconds/3600:.1f} h ({labelled_seconds/record_seconds:.1%}); "
              f"normal `(N` time: {sum(s['seconds'] for s in by_label['(N'])/3600:.1f} h; "
              f"VT/SVTA spans: {len(target)} in {len({s['record'] for s in target})} records.", "",
              "Study implication: this is a long continuous rhythm-labelled source suitable for a streaming protocol pilot, but the AF-enriched sampling frame and record-level concentration of VT/SVTA must be carried into split design and uncertainty estimates. It is not a direct external nine-class VitalDB test.", ""]
    out = ROOT / "outputs/tables/ltafdb_annotation_audit.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(dict(records=len(records), spans=spans), indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
