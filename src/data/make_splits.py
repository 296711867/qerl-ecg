# -*- coding: utf-8 -*-
"""患者级分层分组 5 折划分。

原则（chatGPT建议3 第19节）：
- 分组单位 = case_id（绝不允许同一患者跨 train/test）
- 分层目标 = 9 类 rhythm 的病例在场数尽量均衡（稀有类优先：AVB 仅 10 例 → 每折 2 例）
- 折 i 作为 test 时，折 (i+1)%5 作为 val，其余 3 折为 train

输出 data/splits/folds.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["N", "AFIB/AFL", "AVB", "SND", "SR-mPAC-BT", "SR-mPVC-BT", "SVTA", "VT", "MAT"]
N_FOLDS = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None,
                    help="None reproduces the frozen folds.json; an integer also shuffles "
                         "within-class case order and writes folds_p<seed>.json")
    args = ap.parse_args()
    RNG = np.random.RandomState(42 if args.seed is None else args.seed)
    beats = pd.read_parquet(ROOT / "data/processed/beats.parquet")
    meta = pd.read_csv(ROOT / "data/annotations/metadata.csv")

    presence = {}
    for label, grp in beats[beats.rhythm_label.isin(CLASSES)].groupby("rhythm_label"):
        presence[label] = set(int(c) for c in grp.case_id.unique())
    all_cases = sorted(int(c) for c in meta.case_id.unique())

    # 按稀有度排序（病例数最少的类先分配）
    order = sorted(CLASSES, key=lambda c: len(presence[c]))
    fold_cases = [[] for _ in range(N_FOLDS)]
    fold_class_count = {c: [0] * N_FOLDS for c in CLASSES}
    assigned = set()

    for cls in order:
        members = sorted(presence[cls])
        if args.seed is not None:
            RNG.shuffle(members)
        for cid in members:
            if cid in assigned:
                continue
            # 选该类计数最少、总病例数最少的折
            k = min(range(N_FOLDS),
                    key=lambda f: (fold_class_count[cls][f], len(fold_cases[f])))
            fold_cases[k].append(cid)
            for present_class in CLASSES:
                if cid in presence[present_class]:
                    fold_class_count[present_class][k] += 1
            assigned.add(cid)

    # 剩余病例按折大小均衡分配（带随机抖动）
    rest = [c for c in all_cases if c not in assigned]
    RNG.shuffle(rest)
    for cid in rest:
        k = min(range(N_FOLDS), key=lambda f: len(fold_cases[f]))
        fold_cases[k].append(cid)

    # 校验与统计
    folds = {}
    for i in range(N_FOLDS):
        test_f = fold_cases[i]
        val_f = fold_cases[(i + 1) % N_FOLDS]
        train_f = [c for j in range(N_FOLDS) if j not in (i, (i + 1) % 5)
                   for c in fold_cases[j]]
        folds[f"fold_{i}"] = dict(train=train_f, val=val_f, test=test_f)

    # 分层均衡性报告（按最终折成员重新计数）
    lines = ["# 分层 5 折划分报告\n", "| class | " + " | ".join(f"f{i}" for i in range(N_FOLDS)) + " |",
             "|---" * (N_FOLDS + 1) + "|"]
    fold_sets = [set(f) for f in fold_cases]
    for cls in CLASSES:
        actual = [len(presence[cls] & fold_sets[f]) for f in range(N_FOLDS)]
        lines.append(f"| {cls} | " + " | ".join(map(str, actual)) + " |")
    lines.append("| 折大小 | " + " | ".join(str(len(f)) for f in fold_cases) + " |")
    report = "\n".join(lines)
    (ROOT / "outputs/tables").mkdir(exist_ok=True, parents=True)
    suffix = "" if args.seed is None else f"_p{args.seed}"
    (ROOT / f"outputs/tables/split_report{suffix}.md").write_text(report, encoding="utf-8")

    out = ROOT / f"data/splits/folds{suffix}.json"
    out.write_text(json.dumps(folds, indent=1))
    print(report)
    print(f"\nsaved: {out}  (fold_0: train={len(folds['fold_0']['train'])}, "
          f"val={len(folds['fold_0']['val'])}, test={len(folds['fold_0']['test'])})")


if __name__ == "__main__":
    main()
