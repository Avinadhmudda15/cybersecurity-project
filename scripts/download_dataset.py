"""
Download Edge-IIoTset DNN CSV via Kaggle API into ./data/

Requires kaggle.json in %USERPROFILE%\\.kaggle\\
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / "DNN-EdgeIIoT-dataset.csv"
    if out.is_file():
        print("Already present:", out)
        return 0
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        "mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot",
        "-p",
        str(data_dir),
        "--unzip",
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
