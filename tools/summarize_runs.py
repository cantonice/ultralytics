#!/usr/bin/env python3
"""Summarize Ultralytics experiment runs into a single CSV.

Scans run directories and extracts:
- Core args from args.yaml
- Last epoch metrics
- Best epoch metrics (by mAP50-95)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.utils import YAML


def safe_float(value, default=0.0) -> float:
    """Convert value to float with fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_rows(csv_path: Path) -> list[dict]:
    """Read CSV rows as list of dicts."""
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def summarize_one_run(run_dir: Path) -> dict:
    """Summarize one run directory."""
    args_path = run_dir / "args.yaml"
    results_csv = run_dir / "results.csv"

    args = YAML.load(args_path) if args_path.exists() else {}
    rows = read_rows(results_csv)

    best_row = {}
    last_row = {}
    if rows:
        last_row = rows[-1]
        best_row = max(rows, key=lambda r: safe_float(r.get("metrics/mAP50-95(B)"), default=-1.0))

    return {
        "run_dir": str(run_dir),
        "status": "finished" if rows else "no_results_csv",
        "model": args.get("model", ""),
        "data": args.get("data", ""),
        "epochs": args.get("epochs", ""),
        "batch": args.get("batch", ""),
        "imgsz": args.get("imgsz", ""),
        "device": args.get("device", ""),
        "seed": args.get("seed", ""),
        "name": args.get("name", run_dir.name),
        "project": args.get("project", ""),
        "last_epoch": last_row.get("epoch", ""),
        "last_mAP50": last_row.get("metrics/mAP50(B)", ""),
        "last_mAP50_95": last_row.get("metrics/mAP50-95(B)", ""),
        "last_precision": last_row.get("metrics/precision(B)", ""),
        "last_recall": last_row.get("metrics/recall(B)", ""),
        "best_epoch": best_row.get("epoch", ""),
        "best_mAP50": best_row.get("metrics/mAP50(B)", ""),
        "best_mAP50_95": best_row.get("metrics/mAP50-95(B)", ""),
        "best_precision": best_row.get("metrics/precision(B)", ""),
        "best_recall": best_row.get("metrics/recall(B)", ""),
    }


def discover_runs(runs_root: Path) -> list[Path]:
    """Discover run directories with args.yaml."""
    if not runs_root.exists():
        return []
    return sorted([d for d in runs_root.iterdir() if d.is_dir() and (d / "args.yaml").exists()])


def write_summary(rows: list[dict], output_csv: Path) -> None:
    """Write summary rows to CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_dir",
        "status",
        "model",
        "data",
        "epochs",
        "batch",
        "imgsz",
        "device",
        "seed",
        "name",
        "project",
        "last_epoch",
        "last_mAP50",
        "last_mAP50_95",
        "last_precision",
        "last_recall",
        "best_epoch",
        "best_mAP50",
        "best_mAP50_95",
        "best_precision",
        "best_recall",
    ]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Summarize Ultralytics run results")
    parser.add_argument("--runs-root", type=str, default="runs/sf", help="Root directory for experiment runs")
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/sf/results/summary.csv",
        help="Output summary CSV path",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    runs_root = Path(args.runs_root)
    output = Path(args.output)

    run_dirs = discover_runs(runs_root)
    summaries = [summarize_one_run(d) for d in run_dirs]
    write_summary(summaries, output)

    print(f"[Summary] runs found: {len(run_dirs)}")
    print(f"[Summary] saved to: {output}")


if __name__ == "__main__":
    main()
