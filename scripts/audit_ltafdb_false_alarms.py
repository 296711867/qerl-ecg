"""Validation-only audit of SVTA false-alarm backgrounds and transition distance."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FS = 128


def main():
    records = json.loads((ROOT / "data/splits/ltafdb_record_folds.json").read_text())["fold_0"]["val"]
    spans = defaultdict(list)
    for s in json.loads((ROOT / "outputs/tables/ltafdb_annotation_audit.json").read_text())["spans"]:
        spans[s["record"]].append(s)
    summary = {}
    for tag in ("v1", "neg1000"):
        for threshold in (0.9, 0.999):
            by_rhythm, by_record, interior, hours = Counter(), Counter(), Counter(), Counter()
            for record in records:
                d = np.load(ROOT / f"outputs/stream_pilot/fold0_{tag}_val/{record}.npz")
                t, scores = d["t"], d["p"][:, 1].astype(np.float32)
                points = t.astype(np.int64) * FS
                group = spans[record]
                starts = np.array([s["start_sample"] for s in group], dtype=np.int64)
                ends = np.array([s["end_sample"] for s in group], dtype=np.int64)
                labels = np.array([s["label"] for s in group])
                index = np.searchsorted(starts, points, side="right") - 1
                known = (index >= 0) & (points < ends[np.maximum(index, 0)])
                state = np.where(known, labels[np.maximum(index, 0)], "unknown")
                next_index = np.searchsorted(starts, points)
                before = np.where(next_index > 0, points - starts[np.maximum(next_index - 1, 0)], 10**12)
                after = np.where(next_index < len(starts), starts[np.minimum(next_index, len(starts) - 1)] - points, 10**12)
                distance = np.minimum(before, after)
                eligible = known & (distance > FS) & (state != "(SVTA")
                above = scores >= threshold
                alarms = above & ~np.r_[False, above[:-1]] & eligible
                for rhythm in np.unique(state[eligible]):
                    hours[str(rhythm)] += int(np.sum(eligible & (state == rhythm))) / 3600
                for rhythm in np.unique(state[alarms]):
                    mask = alarms & (state == rhythm)
                    count = int(np.sum(mask))
                    by_rhythm[str(rhythm)] += count
                    interior[str(rhythm)] += int(np.sum(mask & (distance > 5 * FS)))
                    by_record[record] += count
            summary[f"{tag}_{threshold}"] = dict(false_alarms=sum(by_rhythm.values()),
                                                   by_rhythm=dict(by_rhythm),
                                                   by_record=dict(by_record),
                                                   more_than_5s_from_transition=dict(interior),
                                                   non_target_hours_by_rhythm=dict(hours))
    out = ROOT / "outputs/tables/ltafdb_svta_false_alarm_audit.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Fold-0 validation SVTA false-alarm audit", "",
             "Threshold rising-edge alarms in the already cached 1 Hz validation streams; no outer test scores read. Eligible non-SVTA seconds are rhythm-labelled and >1 s from a transition. Interior means >5 s from every transition; the CNN input is the preceding 5 s.", "",
             "| Model | Threshold | Total FP | AFIB FP | AFIB share | AFIB interior FP | AFIB hours | AFIB FP/h | Top 5 records / total FP |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, row in summary.items():
        af = row["by_rhythm"].get("(AFIB", 0)
        interior = row["more_than_5s_from_transition"].get("(AFIB", 0)
        ah = row["non_target_hours_by_rhythm"].get("(AFIB", 0.0)
        top = sorted(row["by_record"].items(), key=lambda item: -item[1])[:5]
        lines.append(f"| {key.rsplit('_', 1)[0]} | {key.rsplit('_', 1)[1]} | {row['false_alarms']} | "
                     f"{af} | {af / row['false_alarms']:.1%} | {interior} | {ah:.1f} | "
                     f"{af / ah:.2f} | {', '.join(f'{r}:{n}' for r, n in top)} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.with_suffix(".md"))


if __name__ == "__main__":
    main()
