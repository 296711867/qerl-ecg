"""Cache 1 Hz causal classifier scores for validation records only."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import wfdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.stream_cnn import StreamCNN

FS, WINDOW = 128, 640


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--records", nargs="*", help="optional validation-record smoke subset")
    args = ap.parse_args()
    split = json.loads((ROOT / "data/splits/ltafdb_record_folds.json").read_text())[f"fold_{args.fold}"]
    records = args.records or split["val"]
    assert set(records) <= set(split["val"]), "Pilot inference is validation-only"
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StreamCNN().to(device)
    checkpoint = torch.load(ROOT / f"outputs/checkpoints/ltafdb_cnn_f{args.fold}_{args.tag}.pt",
                            map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    outdir = ROOT / f"outputs/stream_pilot/fold{args.fold}_{args.tag}_val"
    outdir.mkdir(parents=True, exist_ok=True)
    idx = np.arange(WINDOW)
    for record in records:
        signal = wfdb.rdrecord(str(ROOT / "data/external/ltafdb" / record)).p_signal.astype(np.float32)
        times = np.arange(5, len(signal) // FS + 1, dtype=np.int32)
        probs = np.empty((len(times), 3), np.float16)
        with torch.inference_mode():
            for start in range(0, len(times), 1024):
                batch_times = times[start:start + 1024]
                windows = signal[(batch_times[:, None] * FS - WINDOW) + idx].transpose(0, 2, 1)
                mean = np.nanmean(windows, axis=2, keepdims=True)
                std = np.nanstd(windows, axis=2, keepdims=True)
                windows = np.clip(np.nan_to_num((windows - mean) / np.maximum(std, 1e-4)), -8, 8)
                x = torch.from_numpy(windows).to(device)
                probs[start:start + len(batch_times)] = model(x).softmax(1).cpu().numpy()
        np.savez_compressed(outdir / f"{record}.npz", t=times, p=probs)
        print(f"{record} {len(times)} seconds", flush=True)
    print(f"validation scores saved to {outdir}; outer test untouched", flush=True)


if __name__ == "__main__":
    main()
