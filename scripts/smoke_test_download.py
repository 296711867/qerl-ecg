# -*- coding: utf-8 -*-
"""冒烟测试：下载单例 ECG 波形，验证与标注文件的时间对齐。

验证点：
1. vitaldb API 可访问，能拉到 ECG_II 波形
2. 波形时长覆盖标注的 analysis 窗口
3. R 峰时间戳能落到波形内，且邻近位置确实存在 QRS 波峰
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import vitaldb

CASE_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 1001
FS = 100  # Hz

t0 = time.time()
vf = vitaldb.VitalFile(CASE_ID)
ecg = vf.to_numpy(["ECG_II"], 1 / FS)[:, 0]
dl_sec = time.time() - t0
print(f"[download] case {CASE_ID}: {len(ecg)} samples @ {FS}Hz = {len(ecg)/FS/60:.1f} min, took {dl_sec:.1f}s")
print(f"[nan ratio] {np.isnan(ecg).mean()*100:.2f}%")

# 加载标注
ann = pd.read_csv(f"data/annotations/Annotation_Files_250907/Annotation_file_{CASE_ID}.csv")
meta = pd.read_csv("data/annotations/metadata.csv")
m = meta[meta.case_id == CASE_ID].iloc[0]
print(f"\n[annotation] {len(ann)} rows, analysis window {m.analysis_start_time_sec:.1f} ~ {m.analysis_end_time_sec:.1f} s ({m.analyzed_duration_sec/60:.1f} min)")
print(f"[annotation] rhythm classes: {m.rhythm_classes}")

# 对齐检查：标注时间是否都落在波形范围内
in_range = (ann.time_second >= 0) & (ann.time_second < len(ecg) / FS)
print(f"\n[align] {in_range.sum()}/{len(ann)} beats fall inside waveform range [0, {len(ecg)/FS:.1f}s]")

# R 峰真实性抽查：取 5 个标注 R 峰，检查 ±150ms 内是否有局部最大值（QRS 峰）
valid = ann[in_range & ann.beat_type.isin(["N", "S", "V"])].head(5)
ok = 0
for _, row in valid.iterrows():
    idx = int(row.time_second * FS)
    lo, hi = max(0, idx - 15), min(len(ecg), idx + 15)
    seg = ecg[lo:hi]
    if np.isnan(seg).all():
        continue
    peak_idx = lo + int(np.nanargmax(np.abs(seg)))
    offset_ms = (peak_idx - idx) / FS * 1000
    amp = ecg[peak_idx]
    print(f"  beat t={row.time_second:9.3f}s type={row.beat_type} rhythm={row.rhythm_label:10s} "
          f"nearest peak offset={offset_ms:+6.1f}ms amp={amp:8.3f}")
    if abs(offset_ms) <= 100:
        ok += 1
print(f"\n[align] R-peak spot check: {ok}/{len(valid)} within ±100ms")
