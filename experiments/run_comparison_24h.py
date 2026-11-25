"""Repository-local entry point for the paired T0 versus T0+24h experiment."""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trendit_lab.compare_24h import run_comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two forests on the same 24h-observed events")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "baseline.toml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("comparison_24h_%Y%m%dT%H%M%S%fZ")
    output = args.output or ROOT / "reports" / "runs" / stamp
    run_comparison(ROOT, args.config.resolve(), output.resolve())
