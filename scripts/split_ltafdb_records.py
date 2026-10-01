"""Freeze five record-disjoint outer folds with a separate validation fold."""
import json
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/external/ltafdb"


def main():
    records = (DATA / "RECORDS").read_text().split()
    spans = json.loads((ROOT / "outputs/tables/ltafdb_annotation_audit.json").read_text())["spans"]
    labels = defaultdict(set)
    for span in spans:
        if span["label"] in ("(VT", "(SVTA") and span["seconds"] >= 1:
            labels[span["record"]].add(span["label"])
    strata = ["both" if len(labels[r]) == 2 else
              "vt" if "(VT" in labels[r] else
              "svta" if "(SVTA" in labels[r] else "neither" for r in records]
    fold_ids = [None] * len(records)
    for fold, (_, test_idx) in enumerate(StratifiedKFold(
            n_splits=5, shuffle=True, random_state=42).split(records, strata)):
        for idx in test_idx:
            fold_ids[idx] = fold
    assert all(x is not None for x in fold_ids)
    result = {}
    lines = ["# Frozen LTAFDB record split", "",
             "Five stratified record folds (random seed 42) using only annotation-level VT/SVTA presence for balancing; each outer test fold is held out, the next fold is validation, and the other three are training. This file was generated before training any LTAFDB model or examining its predictions. The public dataset does not provide a verified subject identity map; this is record-disjoint, not a verified patient-disjoint split.", "",
             "| Outer fold | Train records | Validation records | Test records | Test VT-positive | Test SVTA-positive |",
             "|---:|---:|---:|---:|---:|---:|"]
    for fold in range(5):
        parts = {part: [r for r, f in zip(records, fold_ids) if f in ids]
                 for part, ids in (("train", {(fold + 2) % 5, (fold + 3) % 5, (fold + 4) % 5}),
                                   ("val", {(fold + 1) % 5}), ("test", {fold}))}
        assert not (set(parts["train"]) & set(parts["val"]))
        assert not (set(parts["train"]) & set(parts["test"]))
        assert not (set(parts["val"]) & set(parts["test"]))
        assert sum(map(len, parts.values())) == len(records)
        result[f"fold_{fold}"] = parts
        vt = sum("(VT" in labels[r] for r in parts["test"])
        svta = sum("(SVTA" in labels[r] for r in parts["test"])
        assert vt and svta
        lines.append(f"| {fold} | {len(parts['train'])} | {len(parts['val'])} | "
                     f"{len(parts['test'])} | {vt} | {svta} |")
    out = ROOT / "data/splits/ltafdb_record_folds.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (ROOT / "outputs/tables/ltafdb_record_split.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
