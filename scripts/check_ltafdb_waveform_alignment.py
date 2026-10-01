"""Verify three LTAFDB waveform excerpts align with rhythm-transition samples."""
import hashlib
from pathlib import Path

import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/external/ltafdb"


def main():
    expected = {name: digest for digest, name in
                (line.split(maxsplit=1) for line in (DATA / "SHA256SUMS.txt").read_text().splitlines())}
    lines = ["# LTAFDB three-record waveform alignment smoke check", "",
             "Official S3 waveform files 00, 01, 03 were checked against the PhysioNet SHA256 manifest. This checks file/time-axis integrity only; no classifier or clinical onset was validated.", "",
             "| Record | First target | Onset (s) | Lead names | Extracted samples × leads | Finite share | Lead SD around onset |",
             "|---|---|---:|---|---:|---:|---:|"]
    for name in ("00", "01", "03"):
        path = DATA / f"{name}.dat"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[path.name], name
        header = wfdb.rdheader(str(DATA / name))
        ann = wfdb.rdann(str(DATA / name), "atr")
        targets = [(int(s), str(note).strip().strip("\x00"))
                   for s, note in zip(ann.sample, ann.aux_note)
                   if str(note).strip().strip("\x00") in ("(VT", "(SVTA")]
        assert targets, name
        onset, label = targets[0]
        fs = int(header.fs)
        lo, hi = max(0, onset - 5 * fs), min(header.sig_len, onset + 5 * fs)
        signal = wfdb.rdrecord(str(DATA / name), sampfrom=lo, sampto=hi).p_signal
        assert signal.shape == (hi - lo, header.n_sig)
        assert np.isfinite(signal).mean() > 0.99
        sd = np.nanstd(signal, axis=0)
        lines.append(f"| {name} | `{label}` | {onset/fs:.2f} | {', '.join(header.sig_name)} | "
                     f"{signal.shape[0]} × {signal.shape[1]} | {np.isfinite(signal).mean():.3f} | "
                     f"{', '.join(f'{x:.3f}' for x in sd)} |")
    lines += ["", "All three examples passed checksum, annotation-index, sample-count and finite-value checks. This does not establish label accuracy or suitability of the entire dataset for deployment claims.", ""]
    out = ROOT / "outputs/tables/ltafdb_waveform_smoke.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
