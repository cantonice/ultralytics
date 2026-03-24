#!/usr/bin/env python3
"""Unified experiment runner for Ultralytics training.

This script standardizes experiment execution with:
- Declarative experiment configs
- Consistent run naming and project path
- Reproducible defaults
- Config snapshot into each run directory
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from pprint import pformat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils import YAML


def load_cfg(path: str) -> dict:
    """Load experiment config YAML as dictionary."""
    cfg = YAML.load(path)
    if not isinstance(cfg, dict):
        raise ValueError(f"Experiment config must be a dict: {path}")
    return cfg


def resolve_cfg(cfg: dict, args: argparse.Namespace) -> dict:
    """Resolve config with CLI overrides."""
    resolved = copy.deepcopy(cfg)

    if args.model:
        resolved["model"] = args.model
    if args.weights is not None:
        resolved["weights"] = args.weights
    if args.data:
        resolved["data"] = args.data
    if args.project:
        resolved["project"] = args.project
    if args.name:
        resolved["name"] = args.name

    train = resolved.setdefault("train", {})
    if args.epochs is not None:
        train["epochs"] = args.epochs
    if args.batch is not None:
        train["batch"] = args.batch
    if args.imgsz is not None:
        train["imgsz"] = args.imgsz
    if args.device is not None:
        train["device"] = args.device

    train.setdefault("seed", 0)
    train.setdefault("deterministic", True)
    train.setdefault("workers", 8)

    if "model" not in resolved:
        raise ValueError("Missing required field: model")
    if "data" not in resolved:
        raise ValueError("Missing required field: data")

    return resolved


def build_train_kwargs(cfg: dict) -> dict:
    """Build kwargs passed into model.train()."""
    kwargs = dict(cfg.get("train", {}))
    kwargs["data"] = cfg["data"]
    if "project" in cfg:
        kwargs["project"] = cfg["project"]
    if "name" in cfg:
        kwargs["name"] = cfg["name"]
    return kwargs


def snapshot_configs(exp_cfg_path: str, resolved_cfg: dict, save_dir: Path) -> None:
    """Save original and resolved configs for reproducibility."""
    save_dir.mkdir(parents=True, exist_ok=True)
    original = YAML.load(exp_cfg_path)
    YAML.save(save_dir / "exp_config_original.yaml", original)
    YAML.save(save_dir / "exp_config_resolved.yaml", resolved_cfg)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Run a standardized Ultralytics experiment")
    parser.add_argument("--exp", type=str, required=True, help="Path to experiment YAML")
    parser.add_argument("--model", type=str, default=None, help="Override model path")
    parser.add_argument("--weights", type=str, default=None, help="Override weights path; empty disables loading")
    parser.add_argument("--data", type=str, default=None, help="Override data yaml path")
    parser.add_argument("--project", type=str, default=None, help="Override run project directory")
    parser.add_argument("--name", type=str, default=None, help="Override run name")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. 0 or cpu")
    parser.add_argument("--dry-run", action="store_true", help="Only build model and print config")
    parser.add_argument(
        "--dry-run-load-weights",
        action="store_true",
        help="When --dry-run is enabled, still load weights",
    )
    parser.add_argument(
        "--allow-auto-download",
        action="store_true",
        help="Allow missing weight files to be auto-downloaded by Ultralytics",
    )
    return parser.parse_args()


def main() -> None:
    """Run training from experiment config."""
    args = parse_args()
    cfg = load_cfg(args.exp)
    resolved = resolve_cfg(cfg, args)

    model_path = resolved["model"]
    weights = resolved.get("weights", None)

    if weights and (not Path(weights).exists()) and (not args.allow_auto_download):
        raise FileNotFoundError(
            f"Weights not found: {weights}. "
            "Use an existing local path or pass --allow-auto-download."
        )

    should_load_weights = bool(weights) and (not args.dry_run or args.dry_run_load_weights)

    if should_load_weights:
        model = YOLO(model_path).load(weights)
    else:
        model = YOLO(model_path)

    train_kwargs = build_train_kwargs(resolved)

    print("[Experiment] Resolved config:")
    print(pformat(resolved, sort_dicts=False))

    if args.dry_run:
        model.info()
        print("[Experiment] Dry run completed.")
        return

    results = model.train(**train_kwargs)

    save_dir = None
    if hasattr(results, "save_dir") and results.save_dir is not None:
        save_dir = Path(results.save_dir)
    elif getattr(model, "trainer", None) is not None and getattr(model.trainer, "save_dir", None) is not None:
        save_dir = Path(model.trainer.save_dir)

    if save_dir is not None:
        snapshot_configs(args.exp, resolved, save_dir)
        print(f"[Experiment] Config snapshots saved to: {save_dir}")


if __name__ == "__main__":
    main()
