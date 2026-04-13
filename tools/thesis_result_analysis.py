#!/usr/bin/env python3
"""Generate thesis-ready analysis and visualizations for SF ablation runs.

Usage:
    python tools/thesis_result_analysis.py \
        --results-root /root/ultralytics/experiments/sf/results \
        --output-dir /root/ultralytics/experiments/sf/results/thesis_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_MAP = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50_95": "metrics/mAP50-95(B)",
}


# IEEE-friendly, colorblind-safe palette.
IEEE_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#da8bc3"]


def infer_variant(run_name: str) -> str:
    name = run_name.lower()
    if "baseline" in name:
        return "baseline"
    if "v2plus" in name:
        return "v2plus"
    if "parallel_v1" in name or re.search(r"(^|[_-])v1([_-]|$)", name):
        return "v1"
    if "parallel_v2" in name:
        return "v2"
    if "v2" in name:
        return "v2"
    return run_name


def find_runs(results_root: Path) -> list[Path]:
    return sorted(p.parent for p in results_root.glob("*/results.csv") if p.is_file())


def summarize_run(run_dir: Path) -> dict:
    csv_path = run_dir / "results.csv"
    df = pd.read_csv(csv_path)
    df = df.sort_values("epoch").reset_index(drop=True)

    best_idx = df[METRIC_MAP["mAP50_95"]].idxmax()
    best_row = df.loc[best_idx]
    last_row = df.iloc[-1]

    tail_window = min(50, len(df))
    tail_df = df.tail(tail_window)

    out = {
        "run_name": run_dir.name,
        "variant": infer_variant(run_dir.name),
        "epochs_recorded": int(len(df)),
        "best_epoch": int(best_row["epoch"]),
        "train_time_sec_final": float(last_row["time"]),
        "train_time_hr_final": float(last_row["time"]) / 3600.0,
        "last_precision": float(last_row[METRIC_MAP["precision"]]),
        "last_recall": float(last_row[METRIC_MAP["recall"]]),
        "last_mAP50": float(last_row[METRIC_MAP["mAP50"]]),
        "last_mAP50_95": float(last_row[METRIC_MAP["mAP50_95"]]),
        "best_precision_at_best_map": float(best_row[METRIC_MAP["precision"]]),
        "best_recall_at_best_map": float(best_row[METRIC_MAP["recall"]]),
        "best_mAP50_at_best_map": float(best_row[METRIC_MAP["mAP50"]]),
        "best_mAP50_95": float(best_row[METRIC_MAP["mAP50_95"]]),
        "tail_mean_mAP50_95": float(tail_df[METRIC_MAP["mAP50_95"]].mean()),
        "tail_std_mAP50_95": float(tail_df[METRIC_MAP["mAP50_95"]].std(ddof=0)),
    }
    return out


def select_one_per_variant(summary: pd.DataFrame) -> pd.DataFrame:
    # If there are duplicate runs for same variant, keep the best mAP50-95 one.
    selected = (
        summary.sort_values(["variant", "best_mAP50_95"], ascending=[True, False])
        .groupby("variant", as_index=False)
        .head(1)
        .copy()
    )

    order = ["baseline", "v1", "v2", "v2plus"]
    selected["variant_order"] = selected["variant"].apply(lambda x: order.index(x) if x in order else 999)
    selected = selected.sort_values("variant_order").drop(columns=["variant_order"])
    return selected.reset_index(drop=True)


def build_delta_table(selected: pd.DataFrame) -> pd.DataFrame:
    baseline = selected[selected["variant"] == "baseline"]
    if baseline.empty:
        raise ValueError("No baseline run found. Please ensure a baseline directory name contains 'baseline'.")

    b = baseline.iloc[0]
    rows = []
    for _, r in selected.iterrows():
        rows.append(
            {
                "variant": r["variant"],
                "delta_precision_pp": (r["best_precision_at_best_map"] - b["best_precision_at_best_map"]) * 100.0,
                "delta_recall_pp": (r["best_recall_at_best_map"] - b["best_recall_at_best_map"]) * 100.0,
                "delta_mAP50_pp": (r["best_mAP50_at_best_map"] - b["best_mAP50_at_best_map"]) * 100.0,
                "delta_mAP50_95_pp": (r["best_mAP50_95"] - b["best_mAP50_95"]) * 100.0,
                "delta_time_hr": r["train_time_hr_final"] - b["train_time_hr_final"],
            }
        )
    return pd.DataFrame(rows)


def plot_grouped_metrics(selected: pd.DataFrame, out: Path) -> None:
    labels = selected["variant"].tolist()
    data = {
        "Precision": selected["best_precision_at_best_map"].to_numpy(),
        "Recall": selected["best_recall_at_best_map"].to_numpy(),
        "mAP50": selected["best_mAP50_at_best_map"].to_numpy(),
        "mAP50-95": selected["best_mAP50_95"].to_numpy(),
    }

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    for i, (k, vals) in enumerate(data.items()):
        bars = ax.bar(
            x + (i - 1.5) * width,
            vals,
            width,
            label=k,
            color=IEEE_COLORS[i % len(IEEE_COLORS)],
        )
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=13)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ymin = min(v.min() for v in data.values())
    ymax = max(v.max() for v in data.values())
    pad = max(0.01, (ymax - ymin) * 0.2)
    ax.set_ylim(max(0.0, ymin - pad), min(1.0, ymax + pad * 0.5))
    ax.set_ylabel("Metric value")
    ax.set_title("Best-point performance comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(out / "fig1_grouped_metrics.png", bbox_inches="tight")
    fig.savefig(out / "fig1_grouped_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_delta_bar(delta_df: pd.DataFrame, out: Path) -> None:
    data = delta_df[delta_df["variant"] != "baseline"].copy()
    labels = data["variant"].tolist()
    cols = ["delta_precision_pp", "delta_recall_pp", "delta_mAP50_pp", "delta_mAP50_95_pp"]
    col_names = ["Delta P", "Delta R", "Delta mAP50", "Delta mAP50-95"]

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=200)
    for i, c in enumerate(cols):
        bars = ax.bar(
            x + (i - 1.5) * width,
            data[c].to_numpy(),
            width,
            label=col_names[i],
            color=IEEE_COLORS[i % len(IEEE_COLORS)],
        )
        ax.bar_label(bars, fmt="%+.2f", padding=2, fontsize=13)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    all_vals = np.concatenate([data[c].to_numpy() for c in cols])
    vmin = float(np.min(all_vals))
    vmax = float(np.max(all_vals))
    spread = max(0.5, vmax - vmin)
    upper = vmax + spread * 0.2
    # Keep a small area under zero for readability, avoid large empty lower space.
    lower = min(-0.2, vmin - spread * 0.08)
    ax.set_ylim(lower, upper)
    ax.set_ylabel("Delta vs baseline (pp)")
    ax.set_title("Relative gains over baseline")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig2_delta_vs_baseline.png", bbox_inches="tight")
    fig.savefig(out / "fig2_delta_vs_baseline.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_convergence(run_dirs: list[Path], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    for i, run_dir in enumerate(run_dirs):
        df = pd.read_csv(run_dir / "results.csv").sort_values("epoch")
        label = infer_variant(run_dir.name)
        ax.plot(
            df["epoch"],
            df[METRIC_MAP["mAP50_95"]],
            label=label,
            linewidth=1.6,
            color=IEEE_COLORS[i % len(IEEE_COLORS)],
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP50-95")
    ax.set_title("Convergence curve of mAP50-95")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig3_convergence_map5095.png", bbox_inches="tight")
    fig.savefig(out / "fig3_convergence_map5095.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_efficiency(selected: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    x = selected["train_time_hr_final"].to_numpy()
    y = selected["best_mAP50_95"].to_numpy()
    labels = selected["variant"].tolist()

    colors = [IEEE_COLORS[i % len(IEEE_COLORS)] for i in range(len(labels))]
    ax.scatter(x, y, s=80, c=colors)
    for xi, yi, label, color in zip(x, y, labels, colors):
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=13)

    ax.set_xlabel("Training time (hours)")
    ax.set_ylabel("Best mAP50-95")
    ax.set_title("Accuracy-efficiency trade-off")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig4_efficiency_tradeoff.png", bbox_inches="tight")
    fig.savefig(out / "fig4_efficiency_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)


def to_pp(v: float) -> str:
    return f"{v:+.2f}"


def render_markdown_report(selected: pd.DataFrame, delta: pd.DataFrame, out_md: Path) -> None:
    b = selected[selected["variant"] == "baseline"].iloc[0]
    best_map = selected.sort_values("best_mAP50_95", ascending=False).iloc[0]

    lines: list[str] = []
    lines.append("# SF模块消融实验论文级分析报告")
    lines.append("")
    lines.append("## 1. 数据与方法")
    lines.append("- 数据来源：results目录下四组500轮对比实验（baseline/v1/v2/v2plus）。")
    lines.append("- 评价口径：以每组`mAP50-95`最高轮次作为主比较点，同时给出最终轮次与后50轮稳定性统计。")
    lines.append("- 说明：当前为单次训练结果，尚不构成统计显著性结论，建议补充多seed重复实验。")
    lines.append("")
    lines.append("## 2. 核心结论")
    lines.append(
        f"- 最优综合精度来自 **{best_map['variant']}**，其 best mAP50-95 = {best_map['best_mAP50_95']:.4f}。"
    )

    drows = delta.set_index("variant")
    for v in ["v1", "v2", "v2plus"]:
        if v in drows.index:
            r = drows.loc[v]
            lines.append(
                "- "
                + f"{v} 相对 baseline 的变化："
                + f"ΔP={to_pp(r['delta_precision_pp'])} pp, "
                + f"ΔR={to_pp(r['delta_recall_pp'])} pp, "
                + f"ΔmAP50={to_pp(r['delta_mAP50_pp'])} pp, "
                + f"ΔmAP50-95={to_pp(r['delta_mAP50_95_pp'])} pp。"
            )

    lines.append("")
    lines.append("## 3. 学术写作建议（硕士论文）")
    lines.append("- 建议同时报告全量结果与场景子集结果，避免只呈现优势样本导致结论偏置。")
    lines.append("- 对不利指标应给出边界解释（例如precision下降与门控策略敏感性相关）。")
    lines.append("- 最终论文建议补充3~5个随机种子并报告均值±标准差。")
    lines.append("")
    lines.append("## 4. 产出文件")
    lines.append("- summary_selected.csv：每个变体保留一条代表性结果（按best mAP50-95选取）。")
    lines.append("- summary_all_runs.csv：所有run汇总。")
    lines.append("- delta_vs_baseline.csv：相对baseline增益。")
    lines.append("- fig1~fig4：性能对比、增益、收敛、效率权衡图。")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def normalize_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 13,
            "axes.titlesize": 13,
            "axes.labelsize": 13,
            "legend.fontsize": 13,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    normalize_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = find_runs(args.results_root)
    if not run_dirs:
        raise FileNotFoundError(f"No runs found under: {args.results_root}")

    all_rows = [summarize_run(d) for d in run_dirs]
    all_df = pd.DataFrame(all_rows)
    selected = select_one_per_variant(all_df)
    delta = build_delta_table(selected)

    all_df.to_csv(args.output_dir / "summary_all_runs.csv", index=False)
    selected.to_csv(args.output_dir / "summary_selected.csv", index=False)
    delta.to_csv(args.output_dir / "delta_vs_baseline.csv", index=False)

    plot_grouped_metrics(selected, args.output_dir)
    plot_delta_bar(delta, args.output_dir)
    plot_convergence(run_dirs, args.output_dir)
    plot_efficiency(selected, args.output_dir)

    render_markdown_report(selected, delta, args.output_dir / "thesis_analysis_report.md")

    print(f"Analysis completed. Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
