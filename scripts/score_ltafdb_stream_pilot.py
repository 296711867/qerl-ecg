"""Validation-only event recall and false alarms for simple score thresholds."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FS = 128
TAGS = {1: "(SVTA", 2: "(VT"}
THRESHOLDS = (0.5, 0.9, 0.99, 0.999)


def hysteresis_alarms(scores, enter, release):
    alarms = np.zeros(len(scores), dtype=bool)
    active = False
    for i, score in enumerate(scores):
        if active and score < release:
            active = False
        if not active and score >= enter:
            alarms[i] = True
            active = True
    return alarms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--hysteresis", action="store_true", help="release at half the entry threshold")
    args = ap.parse_args()
    split = json.loads((ROOT / "data/splits/ltafdb_record_folds.json").read_text())["fold_0"]
    spans = defaultdict(list)
    for s in json.loads((ROOT / "outputs/tables/ltafdb_annotation_audit.json").read_text())["spans"]:
        spans[s["record"]].append(s)
    totals = {(tag, th): dict(events=0, hits=0, false=0, hours=0.0, repeat=0, delay=[])
              for tag in TAGS for th in THRESHOLDS}
    for record in split["val"]:
        data = np.load(ROOT / f"outputs/stream_pilot/fold0_{args.tag}_val/{record}.npz")
        t, p = data["t"], data["p"].astype(np.float32)
        group = spans[record]
        starts = np.array([s["start_sample"] for s in group], dtype=np.int64)
        ends = np.array([s["end_sample"] for s in group], dtype=np.int64)
        labels = np.array([s["label"] for s in group])
        index = np.searchsorted(starts, t.astype(np.int64) * FS, side="right") - 1
        known = (index >= 0) & (t.astype(np.int64) * FS < ends[np.maximum(index, 0)])
        current = np.where(known, labels[np.maximum(index, 0)], "unknown")
        # Only label-covered seconds at least 1 s from every rhythm transition count toward FA/h.
        pos = np.searchsorted(starts, t.astype(np.int64) * FS)
        before = np.where(pos > 0, t.astype(np.int64) * FS - starts[np.maximum(pos - 1, 0)], 10**12)
        after = np.where(pos < len(starts), starts[np.minimum(pos, len(starts) - 1)] - t.astype(np.int64) * FS, 10**12)
        safe = known & (np.minimum(before, after) > FS)
        for cls, tag in TAGS.items():
            events = [s for s in group if s["label"] == tag and s["seconds"] >= 1]
            for threshold in THRESHOLDS:
                out = totals[(cls, threshold)]
                if args.hysteresis:
                    alarms = hysteresis_alarms(p[:, cls], threshold, threshold / 2)
                else:
                    above = p[:, cls] >= threshold
                    alarms = above & ~np.r_[False, above[:-1]]
                alarm_t = t[alarms].astype(np.int64) * FS
                out["events"] += len(events)
                out["hours"] += float(np.sum(safe & (current != tag))) / 3600
                out["false"] += int(np.sum(alarms & safe & (current != tag)))
                for s in events:
                    hits = alarm_t[(alarm_t > s["start_sample"]) & (alarm_t < s["end_sample"])]
                    if len(hits):
                        out["hits"] += 1
                        out["repeat"] += len(hits) - 1
                        out["delay"].append(float((hits[0] - s["start_sample"]) / FS))
        print(record, flush=True)
    lines = ["# LTAFDB fold-0 validation stream pilot", "",
             "**Exploratory validation only.** Same sampled validation windows selected the CNN checkpoint; these estimates are optimistic. All 17 validation records were scored at 1 Hz end to end. No outer test scores were viewed. The subset of known rhythm time more than 1 s from transitions forms each class-specific false-alarm denominator. There is no independently validated artifact mask.", "",
             f"Policy: {'hysteresis, release at entry/2' if args.hysteresis else 'threshold rising edge'}.", "",
             "| Target | Threshold | Events | Hit | Recall | False alarms | Non-target h | FA/h | Median first-alarm delay s | Repeat alarms |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    serial = {}
    for (cls, th), row in totals.items():
        recall = row["hits"] / row["events"] if row["events"] else float("nan")
        fa = row["false"] / row["hours"] if row["hours"] else float("nan")
        delay = float(np.median(row["delay"])) if row["delay"] else float("nan")
        lines.append(f"| {TAGS[cls]} | {th} | {row['events']} | {row['hits']} | {recall:.3f} | "
                     f"{row['false']} | {row['hours']:.1f} | {fa:.2f} | {delay:.2f} | {row['repeat']} |")
        serial[f"{TAGS[cls]}_{th}"] = dict(**{k: v for k, v in row.items() if k != "delay"},
                                           recall=recall, false_alarms_per_hour=fa,
                                           median_delay_s=delay)
    variant = "hysteresis" if args.hysteresis else "threshold"
    path = ROOT / f"outputs/tables/ltafdb_stream_fold0_{args.tag}_{variant}_val_pilot.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps(serial, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
