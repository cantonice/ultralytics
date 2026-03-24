#!/usr/bin/env python3
"""Run a suite of ablation experiments sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.utils import YAML


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Run ablation suite")
    parser.add_argument(
        "--suite",
        type=str,
        default="experiments/sf/configs/ablation/suite_composition_v1.yaml",
        help="Path to ablation suite yaml",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for all experiments")
    parser.add_argument("--device", type=str, default=None, help="Override device for all experiments")
    parser.add_argument("--batch", type=int, default=None, help="Override batch for all experiments")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size for all experiments")
    parser.add_argument("--dry-run", action="store_true", help="Use dry-run mode for all experiments")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately when one experiment fails")
    return parser.parse_args()


def build_command(exp_cfg: str, args: argparse.Namespace) -> list[str]:
    """Build train_experiment command."""
    cmd = [sys.executable, "tools/train_experiment.py", "--exp", exp_cfg]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.device is not None:
        cmd += ["--device", str(args.device)]
    if args.batch is not None:
        cmd += ["--batch", str(args.batch)]
    if args.imgsz is not None:
        cmd += ["--imgsz", str(args.imgsz)]
    if args.dry_run:
        cmd += ["--dry-run"]
    return cmd


def main() -> None:
    """Run ablation experiments one by one."""
    args = parse_args()
    suite = YAML.load(args.suite)

    experiments = suite.get("experiments", [])
    if not experiments:
        raise ValueError(f"No experiments found in suite: {args.suite}")

    print(f"[Ablation] suite: {suite.get('suite', 'unknown')}")
    print(f"[Ablation] total experiments: {len(experiments)}")

    failed = []
    for idx, item in enumerate(experiments, start=1):
        label = item.get("label", f"exp_{idx}")
        cfg = item["config"]
        cmd = build_command(cfg, args)

        print(f"\n[Ablation] ({idx}/{len(experiments)}) {label}")
        print("[Ablation] command:", " ".join(cmd))

        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            failed.append((label, cfg, result.returncode))
            print(f"[Ablation] failed: {label}, code={result.returncode}")
            if args.stop_on_error:
                break

    print("\n[Ablation] done")
    print(f"[Ablation] failed count: {len(failed)}")
    for label, cfg, code in failed:
        print(f"  - {label} | {cfg} | code={code}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
