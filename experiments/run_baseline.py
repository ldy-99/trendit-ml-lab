"""Repository-local entry point: python experiments/run_baseline.py"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trendit_lab.experiment import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the T0 48h baseline")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "baseline.toml")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or ROOT / "reports" / "runs" / stamp
    run(ROOT, args.config.resolve(), output.resolve())
