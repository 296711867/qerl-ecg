"""Draw two annotation-alignment examples for manual waveform inspection."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/external/ltafdb"


def main():
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), constrained_layout=True)
    for ax, name, target in zip(axes, ("00", "01"), ("(VT", "(SVTA")):
        ann = wfdb.rdann(str(DATA / name), "atr")
        onset = next(int(s) for s, note in zip(ann.sample, ann.aux_note)
                     if str(note).strip().strip("\x00") == target)
        fs = int(ann.fs)
        lo, hi = onset - 3 * fs, onset + 5 * fs
        sig = wfdb.rdrecord(str(DATA / name), sampfrom=lo, sampto=hi).p_signal
        t = (np.arange(len(sig)) + lo - onset) / fs
        ax.plot(t, sig[:, 0], linewidth=0.8, label="lead 1")
        ax.axvline(0, color="red", linewidth=1, label="annotated transition")
        ax.set(title=f"LTAFDB {name}: first {target[1:]} transition", xlabel="Seconds from annotation", ylabel="mV")
        ax.legend(loc="upper right", fontsize=8)
    out = ROOT / "outputs/figures/ltafdb_onset_examples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
