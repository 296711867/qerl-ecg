# -*- coding: utf-8 -*-
"""标注清洗 + 统计报告。

清洗规则（依据对 482 个标注文件的逐一核查）：
1. `bad_signal_quality_label` ∈ {StartN, EndN} 的行 = 坏质量段边界标记（非心拍）
2. `beat_type` ∈ {Start, End} 的行 = 同类边界标记（另一种记录格式）
   → 两类标记都用于重建坏质量区间，并从心拍表剔除
3. `beat_type=P`（起搏/融合拍，仅 7 行）保留在心拍表中，标注为 P
4. `rhythm_label` 为空但 beat_type 有效的行（全库约 1 行）→ 用同拍型邻拍的 rhythm 填充
5. Noise / Unclassifiable 的 rhythm 行保留（用于质量头与鲁棒性实验），主任务在窗口层过滤

输出：
- data/processed/beats.parquet        全部心拍（含 case_id）
- data/processed/badq_intervals.parquet 坏质量区间（来源：标记对 + 逐拍标志连缀）
- outputs/tables/dataset_stats.md     统计报告
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANN_DIR = ROOT / "data/annotations/Annotation_Files_250907"
OUT_DIR = ROOT / "data/processed"
BEAT_TYPES_VALID = {"N", "S", "V", "U", "P"}


def parse_markers(df: pd.DataFrame):
    """从两种标记格式提取坏质量区间 [(start,end),...]。"""
    intervals = []
    # 格式1: bad_signal_quality_label = StartN/EndN（成对）
    if "bad_signal_quality_label" not in df.columns:
        s = e = df.iloc[0:0]
    else:
        s = df[df.bad_signal_quality_label.astype(str).str.match(r"Start\d+$", na=False)]
        e = df[df.bad_signal_quality_label.astype(str).str.match(r"End\d+$", na=False)]
    for _, row in s.iterrows():
        n = str(row.bad_signal_quality_label)[5:]
        match = e[e.bad_signal_quality_label.astype(str) == f"End{n}"]
        if len(match):
            t0, t1 = sorted([row.time_second, match.iloc[0].time_second])
            intervals.append((t0, t1, "marker_pair"))
    # 格式2: beat_type = Start/End（顺序成对）
    m2 = df[df.beat_type.isin(["Start", "End"])].sort_values("time_second")
    open_t = None
    for _, row in m2.iterrows():
        if row.beat_type == "Start":
            open_t = row.time_second
        elif open_t is not None:
            intervals.append((open_t, row.time_second, "beat_marker"))
            open_t = None
    return intervals


def flag_runs_to_intervals(df: pd.DataFrame):
    """把逐拍 bad_signal_quality=True 的连缀段转成区间（与标记互补）。"""
    intervals = []
    open_t = None
    prev_t = None
    for t, bad in zip(df.time_second, df.bad_signal_quality.astype(bool)):
        if bad and open_t is None:
            open_t = t
        elif not bad and open_t is not None:
            if prev_t is not None and prev_t - open_t > 0.5:  # 至少覆盖一拍
                intervals.append((open_t, prev_t, "beat_flag"))
            open_t = None
        prev_t = t
    if open_t is not None and prev_t and prev_t - open_t > 0.5:
        intervals.append((open_t, prev_t, "beat_flag"))
    return intervals


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(ROOT / "data/annotations/metadata.csv")

    all_beats, all_intervals = [], []
    n_marker_pairs = n_beat_markers = 0
    for cid in sorted(meta.case_id):
        df = pd.read_csv(ANN_DIR / f"Annotation_file_{cid}.csv")

        # 心拍表：有效 beat_type，剔除标记行
        beats = df[df.beat_type.isin(BEAT_TYPES_VALID)].copy()
        # 空 rhythm 用同拍型邻拍填充
        if beats.rhythm_label.isna().any():
            beats["rhythm_label"] = beats.groupby("beat_type").rhythm_label.transform(
                lambda s: s.ffill().bfill())
        beats["case_id"] = cid
        all_beats.append(beats)

        # 坏质量区间：标记对（优先）∪ 逐拍标志连缀
        ivs = parse_markers(df)
        n_marker_pairs += sum(1 for i in ivs if i[2] == "marker_pair")
        n_beat_markers += sum(1 for i in ivs if i[2] == "beat_marker")
        ivs += flag_runs_to_intervals(df[df.beat_type.isin(BEAT_TYPES_VALID)])
        for t0, t1, src in ivs:
            all_intervals.append(dict(case_id=cid, start=t0, end=t1, source=src))

    beats = pd.concat(all_beats, ignore_index=True)
    beats["rhythm_label"] = beats.rhythm_label.fillna("Unknown")
    ivals = pd.DataFrame(all_intervals)

    beats.to_parquet(OUT_DIR / "beats.parquet", index=False)
    ivals.to_parquet(OUT_DIR / "badq_intervals.parquet", index=False)

    # ===== 统计报告 =====
    lines = ["# 数据集清洗统计报告\n"]
    lines.append(f"- 心拍总数（N/S/V/U/P）: {len(beats)}")
    lines.append(f"- 坏质量区间: {len(ivals)} 条 "
                 f"(标记对 {n_marker_pairs}, beat标记 {n_beat_markers}, "
                 f"逐拍连缀 {len(ivals)-(n_marker_pairs+n_beat_markers)})")
    ivals["dur"] = ivals.end - ivals.start
    lines.append(f"- 坏质量区间总时长: {ivals.dur.sum()/60:.1f} min, "
                 f"中位时长 {ivals.dur.median():.1f}s\n")

    lines.append("## Rhythm 分布（心拍级）\n")
    lines.append("| rhythm | beats | % | cases | segments(中位时长s) |")
    lines.append("|---|---|---|---|---|")
    g = beats.groupby("rhythm_label")
    stats = []
    for label, grp in g:
        n_seg = 0
        durs = []
        for cid, sub in grp.groupby("case_id"):
            t = sub.time_second.sort_values().values
            breaks = np.where(np.diff(t) > 3)[0]
            n_seg += len(breaks) + 1
        stats.append((label, len(grp), len(grp) / len(beats) * 100, grp.case_id.nunique(), n_seg))
    for label, n, pct, ncase, nseg in sorted(stats, key=lambda x: -x[1]):
        lines.append(f"| {label} | {n} | {pct:.2f}% | {ncase} | {nseg} |")
    lines.append("\n## beat_type 分布\n")
    lines.append(beats.beat_type.value_counts().to_markdown())

    report = "\n".join(lines)
    (ROOT / "outputs/tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs/tables/dataset_stats.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
