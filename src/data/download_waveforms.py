# -*- coding: utf-8 -*-
"""批量下载 VitalDB 波形并缓存为本地 npz。

每例只保留 analysis 窗口 ± margin 的 ECG_II（100Hz float32），
输出 data/raw/case_{id}.npz: {ecg, fs, t0, case_id}
支持断点续传（已存在且校验通过的跳过）。
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import vitaldb

ROOT = Path(__file__).resolve().parents[2]
FS = 100
MARGIN_SEC = 60


def download_case(case_id: int, out_dir: Path) -> str:
    """下载单例并保存。返回 'ok' / 'skip' / 'fail:<reason>'。"""
    out = out_dir / f"case_{case_id}.npz"
    if out.exists():
        try:
            with np.load(out) as z:
                if z["ecg"].size > 0:
                    return "skip"
        except Exception:
            out.unlink(missing_ok=True)

    meta = pd.read_csv(ROOT / "data/annotations/metadata.csv")
    m = meta[meta.case_id == case_id]
    if m.empty:
        return "fail:no_metadata"
    m = m.iloc[0]

    for attempt in range(3):
        try:
            vf = vitaldb.VitalFile(case_id)
            ecg = vf.to_numpy(["ECG_II"], 1 / FS)[:, 0].astype(np.float32)
            break
        except Exception as e:
            if attempt == 2:
                return f"fail:download({type(e).__name__}:{e})"
            time.sleep(5 * (attempt + 1))

    total_sec = len(ecg) / FS
    if total_sec < m.analysis_end_time_sec:
        return f"fail:waveform_too_short({total_sec:.0f}s < end {m.analysis_end_time_sec:.0f}s)"

    i0 = max(0, int((m.analysis_start_time_sec - MARGIN_SEC) * FS))
    i1 = min(len(ecg), int((m.analysis_end_time_sec + MARGIN_SEC) * FS))
    seg = ecg[i0:i1]
    if np.isnan(seg).all():
        return "fail:all_nan"

    np.savez_compressed(out, ecg=seg, fs=FS, t0=i0 / FS, case_id=case_id)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 例（0=全部）")
    ap.add_argument("--cases", type=str, default="", help="逗号分隔的指定 case_id")
    args = ap.parse_args()

    meta = pd.read_csv(ROOT / "data/annotations/metadata.csv")
    case_ids = sorted(meta.case_id.tolist())
    if args.cases:
        case_ids = [int(c) for c in args.cases.split(",")]
    elif args.limit:
        case_ids = case_ids[: args.limit]

    out_dir = ROOT / "data/raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = 0
    failures = []
    t_start = time.time()
    for i, cid in enumerate(case_ids, 1):
        status = download_case(cid, out_dir)
        if status == "ok":
            n_ok += 1
        elif status == "skip":
            n_skip += 1
        else:
            failures.append((cid, status))
            print(f"[FAIL] case {cid}: {status}", flush=True)
        if i % 20 == 0 or i == len(case_ids):
            rate = (time.time() - t_start) / max(i - n_skip, 1)
            print(f"[{i}/{len(case_ids)}] ok={n_ok} skip={n_skip} fail={len(failures)} "
                  f"({rate:.1f}s/new-case)", flush=True)

    print(f"\nDone: ok={n_ok} skip={n_skip} fail={len(failures)} in {(time.time()-t_start)/60:.1f} min")
    if failures:
        print("Failed cases:")
        for cid, s in failures:
            print(f"  {cid}: {s}")


if __name__ == "__main__":
    main()
