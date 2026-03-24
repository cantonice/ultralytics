#!/usr/bin/env python3
"""Analyze ablation summary and generate comparison tables/reports."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.utils import YAML


METRICS = [
    "best_mAP50_95",
    "best_mAP50",
    "best_precision",
    "best_recall",
]


def to_float(x: str) -> float:
    """Convert to float safely."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_csv(path: Path) -> list[dict]:
    """Load csv rows."""
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {path}")
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def load_suite(path: Path) -> list[tuple[str, str]]:
    """Load suite labels and expected run names from exp configs."""
    suite = YAML.load(path)
    items = []
    for exp in suite.get("experiments", []):
        label = exp["label"]
        cfg_path = ROOT / exp["config"]
        cfg = YAML.load(cfg_path)
        run_name = cfg.get("name", "")
        items.append((label, run_name))
    return items


def pick_rows(summary_rows: list[dict], suite_items: list[tuple[str, str]]) -> list[dict]:
    """Pick rows by run name based on suite config names."""
    by_name = {r.get("name", ""): r for r in summary_rows}
    selected = []
    for label, run_name in suite_items:
        row = dict(by_name.get(run_name, {}))
        row["label"] = label
        row["run_name"] = run_name
        selected.append(row)
    return selected


def write_table(rows: list[dict], out_csv: Path) -> None:
    """Write compact table."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["label", "run_name", "status"] + METRICS
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def build_report(rows: list[dict], out_md: Path) -> None:
    """Write markdown report with delta vs baseline."""
    out_md.parent.mkdir(parents=True, exist_ok=True)

    baseline = None
    for r in rows:
        if r.get("label") == "baseline":
            baseline = r
            break

    lines = []
    lines.append("# 消融分析报告（composition_v1）")
    lines.append("")

    if baseline is None:
        lines.append("未找到 baseline 行，无法计算增益。")
    else:
        lines.append("## 相对 Baseline 的指标增益")
        lines.append("")
        lines.append("| 变体 | best_mAP50-95 | ΔmAP50-95 | best_mAP50 | ΔmAP50 | best_P | ΔP | best_R | ΔR |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

        b_m95 = to_float(baseline.get("best_mAP50_95", ""))
        b_m50 = to_float(baseline.get("best_mAP50", ""))
        b_p = to_float(baseline.get("best_precision", ""))
        b_r = to_float(baseline.get("best_recall", ""))

        for r in rows:
            m95 = to_float(r.get("best_mAP50_95", ""))
            m50 = to_float(r.get("best_mAP50", ""))
            p = to_float(r.get("best_precision", ""))
            rr = to_float(r.get("best_recall", ""))

            dm95 = m95 - b_m95 if m95 == m95 and b_m95 == b_m95 else float("nan")
            dm50 = m50 - b_m50 if m50 == m50 and b_m50 == b_m50 else float("nan")
            dp = p - b_p if p == p and b_p == b_p else float("nan")
            dr = rr - b_r if rr == rr and b_r == b_r else float("nan")

            lines.append(
                f"| {r.get('label','')} | {m95:.4f} | {dm95:+.4f} | {m50:.4f} | {dm50:+.4f} | {p:.4f} | {dp:+.4f} | {rr:.4f} | {dr:+.4f} |"
            )

        lines.append("")
        lines.append("## 解释建议")
        lines.append("")
        lines.append("- 若 parallel_both 的 ΔmAP50-95 为正，说明空频并行在整体定位与分类综合指标上有效。")
        lines.append("- 若 freq_only 的 Recall 提升而 Precision 下降，说明频域分支偏向召回增益。")
        lines.append("- 若 spatial_only 接近 baseline，而 parallel_both 明显更优，说明创新点来自分支互补而非简单替换。")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Analyze ablation results")
    parser.add_argument(
        "--suite",
        type=str,
        default="experiments/sf/configs/ablation/suite_composition_v1.yaml",
        help="Ablation suite yaml",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="experiments/sf/results/summary.csv",
        help="Summary CSV path produced by summarize_runs.py",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="experiments/sf/results/ablation/composition_v1_table.csv",
        help="Output compact CSV",
    )
    parser.add_argument(
        "--out-report",
        type=str,
        default="experiments/sf/reports/composition_v1_report.md",
        help="Output markdown report",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    suite_items = load_suite(Path(args.suite))
    summary_rows = load_csv(Path(args.summary))
    rows = pick_rows(summary_rows, suite_items)

    write_table(rows, Path(args.out_csv))
    build_report(rows, Path(args.out_report))

    print(f"[Analyze] rows selected: {len(rows)}")
    print(f"[Analyze] table: {args.out_csv}")
    print(f"[Analyze] report: {args.out_report}")


if __name__ == "__main__":
    main()
