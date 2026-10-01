# -*- coding: utf-8 -*-
"""One causal, forward-observed episode per annotated rhythm segment."""
from pathlib import Path

import numpy as np
import pandas as pd

from build_windows import CLASSES, merge_intervals, overlap

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/processed/event_episodes_v2.parquet"


def build():
    segments = pd.read_parquet(ROOT / "data/processed/segments.parquet")
    badq = pd.read_parquet(ROOT / "data/processed/badq_intervals.parquet")
    bad_map = {cid: list(zip(g.start, g.end)) for cid, g in badq.groupby("case_id")}
    noise_map = {cid: list(zip(g.t_start, g.t_end))
                 for cid, g in segments[segments.rhythm == "Noise"].groupby("case_id")}
    quality_map = {cid: merge_intervals(bad_map.get(cid, []) + noise_map.get(cid, []))
                   for cid in segments.case_id.unique()}
    rows = []
    for seg in segments.itertuples():
        if seg.rhythm not in CLASSES:
            continue
        wmax = min(10, int(np.floor(seg.t_end - seg.t_start)))
        if wmax < 1:
            continue
        row = dict(case_id=int(seg.case_id), label=seg.rhythm,
                   t_start=float(seg.t_start), t_segment_end=float(seg.t_end),
                   w_max=wmax, event_id=len(rows))
        ivs = quality_map[seg.case_id]
        for w in range(1, 11):
            row[f"quality_{w}"] = (min(1.0, overlap(seg.t_start, seg.t_start + w, ivs) / w)
                                    if w <= wmax else np.nan)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    print(f"events={len(out)} patients={out.case_id.nunique()} output={OUT}")
    print(out.groupby("label").agg(events=("event_id", "size"),
                                    patients=("case_id", "nunique"),
                                    max_window_median=("w_max", "median")).to_string())
    print(f"quality range: {out.filter(regex='^quality_').min().min():.3f}.."
          f"{out.filter(regex='^quality_').max().max():.3f}")


if __name__ == "__main__":
    build()
