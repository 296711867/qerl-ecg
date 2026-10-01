# -*- coding: utf-8 -*-
"""Causal ECG windows growing forward from an annotated rhythm-segment start."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfilt
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["N", "AFIB/AFL", "AVB", "SND", "SR-mPAC-BT", "SR-mPVC-BT",
           "SVTA", "VT", "MAT"]
FS = 100
MAX_W = 10
SOS = butter(4, [0.5, 40], btype="band", fs=FS, output="sos")


class CausalCaseCache:
    def __init__(self):
        self.cache = {}

    def get(self, case_id):
        if case_id not in self.cache:
            with np.load(ROOT / f"data/raw/case_{case_id}.npz") as z:
                ecg = z["ecg"].astype(np.float64)
                t0 = float(z["t0"])
            if np.isnan(ecg).any():
                good = ~np.isnan(ecg)
                last = np.maximum.accumulate(np.where(good, np.arange(len(ecg)), 0))
                ecg = np.where(good, ecg, ecg[last])
                ecg = np.nan_to_num(ecg)
            self.cache[case_id] = (sosfilt(SOS, ecg).astype(np.float32), t0)
        return self.cache[case_id]


CACHE = CausalCaseCache()


def load_events(fold, part, split="folds"):
    import json

    folds = json.loads((ROOT / f"data/splits/{split}.json").read_text())
    cases = set(folds[f"fold_{fold}"][part])
    events = pd.read_parquet(ROOT / "data/processed/event_episodes_v2.parquet")
    return events[events.case_id.isin(cases)].reset_index(drop=True)


class EventDataset(Dataset):
    """fixed_w=None samples a prefix; fixed_w=0 uses each event's longest prefix."""

    def __init__(self, events, fixed_w=None):
        self.df = (events[events.w_max >= fixed_w].reset_index(drop=True)
                   if fixed_w and fixed_w > 0 else events.reset_index(drop=True))
        self.fixed_w = fixed_w

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        w = (min(int(row.w_max), MAX_W) if self.fixed_w == 0 else
             int(self.fixed_w) if self.fixed_w else np.random.randint(1, int(row.w_max) + 1))
        ecg, t0 = CACHE.get(int(row.case_id))
        start = int(round((row.t_start - t0) * FS))
        signal = ecg[max(start, 0):max(start, 0) + w * FS]
        if len(signal) < w * FS:
            signal = np.pad(signal, (0, w * FS - len(signal)))
        signal = (signal - signal.mean()) / (signal.std() + 1e-6)
        x = np.zeros(MAX_W * FS, dtype=np.float32)
        x[-len(signal):] = signal
        return (torch.from_numpy(x).unsqueeze(0), CLASSES.index(row.label),
                float(row[f"quality_{w}"]), int(row.event_id))
