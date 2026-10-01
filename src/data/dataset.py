# -*- coding: utf-8 -*-
"""ECG 窗口 Dataset：滤波预处理 + 全量内存缓存 + 变长窗口抽取。

预处理（依据 configs/project.yaml）：
- NaN 线性插补
- 带通 0.5–40 Hz（butterworth 4阶, filtfilt）
- 60 Hz 陷波（保留真实噪声，不做强伪影去除）
- 逐窗 z-score
"""
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["N", "AFIB/AFL", "AVB", "SND", "SR-mPAC-BT", "SR-mPVC-BT", "SVTA", "VT", "MAT"]
SEVERE = {"VT", "SVTA", "AVB", "SND"}
FS = 100
RAW_DIR = ROOT / "data/raw"


class EcgCaseCache:
    """全部病例的滤波后波形缓存（~482 例 × ~20min×100Hz ≈ 230MB，可全内存）。"""

    def __init__(self):
        self._cache: dict[int, tuple[np.ndarray, float]] = {}
        self._lock = threading.Lock()
        b, a = butter(4, [0.5 / (FS / 2), 40.0 / (FS / 2)], btype="band")
        # 60Hz 工频在 100Hz 采样下高于奈奎斯特频率，0.5-40Hz 带通已覆盖，无需陷波
        self._ba = (b, a)

    def get(self, case_id: int):
        if case_id in self._cache:
            return self._cache[case_id]
        with self._lock:
            if case_id in self._cache:
                return self._cache[case_id]
            with np.load(RAW_DIR / f"case_{case_id}.npz") as z:
                ecg = z["ecg"].astype(np.float64)
                t0 = float(z["t0"])
            # NaN 插补
            if np.isnan(ecg).any():
                idx = np.arange(len(ecg))
                good = ~np.isnan(ecg)
                if good.sum() < 100:
                    raise ValueError(f"case {case_id}: too many NaN")
                ecg = np.interp(idx, idx[good], ecg[good])
            ecg = filtfilt(*self._ba, ecg)
            self._cache[case_id] = (ecg.astype(np.float32), t0)
            return self._cache[case_id]


CACHE = EcgCaseCache()


class EcgWindowDataset(Dataset):
    """end-anchored 窗口数据集。

    参数:
        win_df: windows.parquet 的行（已按 split 过滤）
        window_sec: 本数据集使用的观察窗长度（1–10s），窗口 = [t_end-w, t_end]
        task: '9class' | 'binary'（Normal vs Abnormal）| 'severe'（severe vs rest）
        drop_badq: 丢弃 quality_frac>0.5 的窗口（监督主实验用；质量实验保留）
    """

    def __init__(self, win_df: pd.DataFrame, window_sec: float, task: str = "9class",
                 drop_badq: bool = False):
        df = win_df[win_df.w_max >= window_sec - 1e-6].copy()
        if drop_badq:
            df = df[df.quality_frac <= 0.5]
        self.df = df.reset_index(drop=True)
        self.w = int(window_sec * FS)
        self.task = task

    def __len__(self):
        return len(self.df)

    def label_of(self, row) -> int:
        if self.task == "binary":
            return 0 if row.label == "N" else 1
        if self.task == "severe":
            return int(row.label in SEVERE)
        return CLASSES.index(row.label)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        ecg, t0 = CACHE.get(int(row.case_id))
        i1 = int(round((row.t_end - t0) * FS))
        i0 = i1 - self.w
        x = ecg[max(0, i0):i1]
        if len(x) < self.w:  # 波形起点余量不足时左补零
            x = np.concatenate([np.zeros(self.w - len(x), dtype=np.float32), x])
        x = x - x.mean()
        std = x.std()
        x = x / (std + 1e-6)
        return (torch.from_numpy(x.copy()).float().unsqueeze(0),
                int(self.label_of(row)),
                float(row.quality_frac),
                int(row.case_id))


def load_split(fold: int = 0, part: str = "train"):
    import json
    folds = json.loads((ROOT / "data/splits/folds.json").read_text())
    case_ids = set(folds[f"fold_{fold}"][part])
    win = pd.read_parquet(ROOT / "data/processed/windows.parquet")
    return win[win.case_id.isin(case_ids)].copy()


def class_counts(task: str, df: pd.DataFrame):
    """返回各类样本数（用于 class-balanced 权重）。"""
    if task == "binary":
        labels = (df.label != "N").astype(int)
    elif task == "severe":
        labels = df.label.isin(SEVERE).astype(int)
    else:
        labels = df.label.map({c: i for i, c in enumerate(CLASSES)})
    return np.bincount(labels.values, minlength=len(CLASSES) if task == "9class" else 2)
