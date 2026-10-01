"""Build ECG-only causal 5-second training windows from LTAFDB records."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/external/ltafdb"
FS = 128
WINDOW = 5 * FS


def times_for_record(spans, rng, negatives_per_record):
    positive, negative_ranges = [], []
    for span in spans:
        start, end = span["start_sample"] / FS, span["end_sample"] / FS
        if span["label"] in ("(VT", "(SVTA") and end - start >= 1:
            label = 2 if span["label"] == "(VT" else 1
            positive.extend((t, label) for t in range(max(WINDOW // FS, int(np.floor(start)) + 1),
                                                        int(np.ceil(end))))
        elif span["label"] not in ("(VT", "(SVTA"):
            a, b = max(WINDOW // FS, int(np.ceil(start + 1))), int(np.ceil(end - 1))
            if b > a:
                negative_ranges.append((a, b))
    counts = np.array([b - a for a, b in negative_ranges], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(counts)]
    picks = rng.choice(offsets[-1], size=min(negatives_per_record, offsets[-1]), replace=False) if offsets[-1] else []
    negative = [(negative_ranges[int(i)][0] + int(p - offsets[int(i)]), 0)
                for p in picks for i in [np.searchsorted(offsets, p, side="right") - 1]]
    return positive + negative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="only process records 00, 01, 03")
    parser.add_argument("--negatives-per-record", type=int, default=200)
    args = parser.parse_args()
    records = ["00", "01", "03"] if args.smoke else (DATA / "RECORDS").read_text().split()
    all_spans = json.loads((ROOT / "outputs/tables/ltafdb_annotation_audit.json").read_text())["spans"]
    by_record = defaultdict(list)
    for span in all_spans:
        by_record[span["record"]].append(span)
    rng = np.random.default_rng(42)
    x, y, rec_ids, times = [], [], [], []
    skipped = 0
    for record in records:
        signal = wfdb.rdrecord(str(DATA / record)).p_signal.astype(np.float32)
        assert signal.shape[1] == 2 and wfdb.rdheader(str(DATA / record)).fs == FS
        for t, label in times_for_record(by_record[record], rng, args.negatives_per_record):
            right = t * FS
            window = signal[right - WINDOW:right]
            if window.shape != (WINDOW, 2) or np.isfinite(window).mean() < 0.99:
                skipped += 1
                continue
            window = np.nan_to_num(window)
            sd = window.std(axis=0)
            if (sd < 1e-4).any():
                skipped += 1
                continue
            window = np.clip((window - window.mean(axis=0)) / sd, -8, 8)
            x.append(window.T.astype(np.float16))
            y.append(label)
            rec_ids.append(record)
            times.append(t)
        print(record, len(x), flush=True)
    suffix = "smoke" if args.smoke else ("v1" if args.negatives_per_record == 200
                                        else f"neg{args.negatives_per_record}")
    out = ROOT / f"data/processed/ltafdb_windows_{suffix}.npz"
    np.savez_compressed(out, x=np.stack(x), y=np.array(y, dtype=np.uint8),
                        record=np.array(rec_ids), t=np.array(times, dtype=np.int32))
    labels, counts = np.unique(y, return_counts=True)
    print(f"{out} windows={len(x)} skipped={skipped} labels={dict(zip(labels.tolist(), counts.tolist()))}")


if __name__ == "__main__":
    main()
