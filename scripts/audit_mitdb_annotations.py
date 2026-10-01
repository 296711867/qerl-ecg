"""Audit official MIT-BIH rhythm transitions from local PhysioNet annotations."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/external/mitdb"
TARGET = ("(VT", "(SVTA")


def main():
    records = (DATA / "RECORDS").read_text().split()
    expected = {name: digest for digest, name in
                (line.split(maxsplit=1) for line in (DATA / "SHA256SUMS.txt").read_text().splitlines())}
    verified = 0
    spans = []
    for record in records:
        for ext in ("atr", "hea"):
            name = f"{record}.{ext}"
            actual = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
            if actual != expected[name]:
                raise ValueError(f"PhysioNet checksum mismatch: {name}")
            verified += 1
        header = (DATA / f"{record}.hea").read_text().splitlines()[0].split()
        fs, n_samples = float(header[2]), int(header[3])
        ann = wfdb.rdann(str(DATA / record), "atr")
        marks = [(int(sample), str(note).strip().strip("\x00"))
                 for sample, note in zip(ann.sample, ann.aux_note)
                 if str(note).strip().startswith("(")]
        assert marks and marks[0][0] < n_samples, record
        for (start, label), (end, _) in zip(marks, marks[1:] + [(n_samples, "END")]):
            if end > start:
                spans.append(dict(record=record, label=label, start_sample=start,
                                  end_sample=end, seconds=(end - start) / fs))
    by_label = defaultdict(list)
    for span in spans:
        by_label[span["label"]].append(span)
    lines = ["# MIT-BIH rhythm annotation feasibility audit", "",
             f"Source: [PhysioNet MIT-BIH v1.0.0](https://physionet.org/content/mitdb/1.0.0/), downloaded from the official anonymous S3 bucket. Verified {verified} .atr/.hea files against the dataset SHA256SUMS.txt. No waveform files were downloaded; no model was trained.", "",
             "The 48 half-hour excerpts come from 47 subjects; records 201 and 202 belong to the same person and must remain in one split. These are selected Holter excerpts, not complete long-term monitoring records.", "",
             "Rhythm spans run from a `+` rhythm transition annotation to the next rhythm transition or the record end. First-marker prefixes are excluded. These retrospective labels define evaluation intervals, not knowledge supplied to a streaming model.", "",
             "| Rhythm tag | Spans | Records | Total minutes | Median span (s) | Spans <1 s |",
             "|---|---:|---:|---:|---:|---:|"]
    for tag in sorted(by_label, key=lambda x: -len(by_label[x])):
        group = by_label[tag]
        durations = np.array([s["seconds"] for s in group])
        lines.append(f"| `{tag}` | {len(group)} | {len({s['record'] for s in group})} | "
                     f"{durations.sum()/60:.2f} | {np.median(durations):.2f} | {int((durations < 1).sum())} |")
    target = [s for s in spans if s["label"] in TARGET]
    known = sum(s["seconds"] for s in spans)
    normal = sum(s["seconds"] for s in by_label["(N"])
    lines += ["", f"All annotated-transition spans: {known/3600:.2f} h; explicitly normal (`(N`) spans: {normal/3600:.2f} h. Target VT/SVTA spans: {len(target)} across {len({s['record'] for s in target})} records. Counts are rhythm transitions, not patient-independent events.", "",
              "Decision: this is adequate for a parsing and continuous-scan prototype, but the small number of target-bearing records limits patient-level uncertainty estimates. Freeze event matching, transition tolerance and the evaluable negative interval definition before scoring alarms; do not call this external validation of VitalDB's nine classes.", ""]
    out = ROOT / "outputs/tables/mitdb_annotation_audit.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(dict(records=len(records), verified_files=verified,
        rhythm_spans=spans), indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
