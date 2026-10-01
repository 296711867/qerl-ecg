"""Fetch and SHA256-check open LTAFDB files from official PhysioNet S3."""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/external/ltafdb"
BASE = "https://physionet-open.s3.amazonaws.com/ltafdb/1.0.0"


def fetch(item):
    name, digest = item
    path = DEST / name
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
        return name
    for attempt in range(3):
        try:
            response = requests.get(f"{BASE}/{name}", timeout=60, stream=True)
            response.raise_for_status()
            tmp = path.with_suffix(path.suffix + ".part")
            checksum = hashlib.sha256()
            with tmp.open("wb") as out:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        out.write(chunk)
                        checksum.update(chunk)
            if checksum.hexdigest() != digest:
                raise ValueError(f"SHA256 mismatch: {name}")
            tmp.replace(path)
            return name
        except requests.RequestException:
            if attempt == 2:
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--waveforms", action="store_true", help="download all ~3.4 GB .dat waveforms")
    args = parser.parse_args()
    records = (DEST / "RECORDS").read_text().split()
    expected = {name: digest for digest, name in
                (line.split(maxsplit=1) for line in (DEST / "SHA256SUMS.txt").read_text().splitlines())}
    extensions = ("dat",) if args.waveforms else ("atr", "hea")
    items = [(f"{record}.{ext}", expected[f"{record}.{ext}"])
             for record in records for ext in extensions]
    with ThreadPoolExecutor(max_workers=4 if args.waveforms else 8) as pool:
        done = 0
        for name in pool.map(fetch, items):
            done += 1
            print(f"{done}/{len(items)} {name}", flush=True)
    print(f"Verified {done} official PhysioNet files for {len(records)} records")


if __name__ == "__main__":
    main()
