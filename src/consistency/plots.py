from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.core import config

plt.style.use("ggplot")


def _read_summary(path: os.PathLike | str) -> pd.Series:
    df = pd.read_csv(path, index_col=0)
    if "mean" in df.index:
        return df.loc["mean"]
    return df.mean(numeric_only=True)


def _bar(ax, labels, values, title, ylim=(0, 1), colors=None):
    x = np.arange(len(labels))
    colors = colors or ["#4C78A8", "#F58518"]
    ax.bar(x, values, color=colors[: len(values)], width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    for xi, v in zip(x, values):
        ax.text(
            xi,
            v + (ylim[1] - ylim[0]) * 0.02,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _violin(ax, data_groups, labels, title, ylim=(0, 1), colors=None):
    parts = ax.violinplot(data_groups, showmeans=True, showmedians=False, showextrema=False)
    colors = colors or ["#4C78A8", "#F58518"]
    for body, c in zip(parts["bodies"], colors[: len(data_groups)]):
        body.set_facecolor(c)
        body.set_edgecolor("black")
        body.set_alpha(0.6)
    if "cmeans" in parts:
        parts["cmeans"].set_color("black")
        parts["cmeans"].set_linewidth(1.5)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylim(*ylim)
    ax.set_title(title)


def plot_detection():
    summary_orig = _read_summary(config.TABLES_DIR / "detection_consistency_orig_summary.csv")
    summary_mv = _read_summary(config.TABLES_DIR / "detection_consistency_multiviews_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _bar(
        axes[0],
        ["orig", "multiviews"],
        [summary_orig["coverage"], summary_mv["coverage"]],
        "Detection coverage (matched / max(count_a,count_b))",
        ylim=(0, 1),
    )
    _bar(
        axes[1],
        ["orig", "multiviews"],
        [summary_orig["mean_iou"], summary_mv["mean_iou"]],
        "Detection mean IoU (matched boxes only)",
        ylim=(0, 1),
    )
    fig.suptitle("Detection consistency overview")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(config.FIGURES_DIR / "consistency_detection_bars.png", dpi=150)
    plt.close(fig)

    d_orig = pd.read_csv(config.TABLES_DIR / "detection_consistency_orig.csv")
    d_mv = pd.read_csv(config.TABLES_DIR / "detection_consistency_multiviews.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _violin(
        axes[0],
        [d_orig["coverage"].values, d_mv["coverage"].values],
        ["orig", "multiviews"],
        "Coverage distribution",
        ylim=(0, 1),
    )
    _violin(
        axes[1],
        [d_orig["mean_iou"].values, d_mv["mean_iou"].values],
        ["orig", "multiviews"],
        "Mean IoU distribution",
        ylim=(0, 1),
    )
    fig.suptitle("Detection consistency distributions")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(config.FIGURES_DIR / "consistency_detection_violin.png", dpi=150)
    plt.close(fig)


def plot_segmentation():
    summary_orig = _read_summary(config.TABLES_DIR / "segmentation_consistency_orig_summary.csv")
    summary_mv = _read_summary(config.TABLES_DIR / "segmentation_consistency_multiviews_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _bar(
        axes[0],
        ["orig", "multiviews"],
        [summary_orig["pixel_agreement"], summary_mv["pixel_agreement"]],
        "Segmentation pixel agreement",
        ylim=(0.5, 1.0),
        colors=["#54A24B", "#B279A2"],
    )
    _bar(
        axes[1],
        ["orig", "multiviews"],
        [summary_orig["miou_proxy"], summary_mv["miou_proxy"]],
        "Segmentation mIoU proxy",
        ylim=(0.0, 1.0),
        colors=["#54A24B", "#B279A2"],
    )
    fig.suptitle("Segmentation consistency overview")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(config.FIGURES_DIR / "consistency_segmentation_bars.png", dpi=150)
    plt.close(fig)

    d_orig = pd.read_csv(config.TABLES_DIR / "segmentation_consistency_orig.csv")
    d_mv = pd.read_csv(config.TABLES_DIR / "segmentation_consistency_multiviews.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _violin(
        axes[0],
        [d_orig["pixel_agreement"].values, d_mv["pixel_agreement"].values],
        ["orig", "multiviews"],
        "Pixel agreement distribution",
        ylim=(0.5, 1.0),
        colors=["#54A24B", "#B279A2"],
    )
    _violin(
        axes[1],
        [d_orig["miou_proxy"].values, d_mv["miou_proxy"].values],
        ["orig", "multiviews"],
        "mIoU proxy distribution",
        ylim=(0.0, 1.0),
        colors=["#54A24B", "#B279A2"],
    )
    fig.suptitle("Segmentation consistency distributions")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(config.FIGURES_DIR / "consistency_segmentation_violin.png", dpi=150)
    plt.close(fig)


def plot_overview_panel():
    d_orig = _read_summary(config.TABLES_DIR / "detection_consistency_orig_summary.csv")
    d_mv = _read_summary(config.TABLES_DIR / "detection_consistency_multiviews_summary.csv")
    s_orig = _read_summary(config.TABLES_DIR / "segmentation_consistency_orig_summary.csv")
    s_mv = _read_summary(config.TABLES_DIR / "segmentation_consistency_multiviews_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    _bar(axes[0, 0], ["orig", "multiviews"], [d_orig["coverage"], d_mv["coverage"]], "Detection coverage", ylim=(0, 1))
    _bar(axes[0, 1], ["orig", "multiviews"], [d_orig["mean_iou"], d_mv["mean_iou"]], "Detection mean IoU", ylim=(0, 1))
    _bar(
        axes[1, 0],
        ["orig", "multiviews"],
        [s_orig["pixel_agreement"], s_mv["pixel_agreement"]],
        "Seg. pixel agreement",
        ylim=(0.5, 1.0),
        colors=["#54A24B", "#B279A2"],
    )
    _bar(
        axes[1, 1],
        ["orig", "multiviews"],
        [s_orig["miou_proxy"], s_mv["miou_proxy"]],
        "Seg. mIoU proxy",
        ylim=(0.0, 1.0),
        colors=["#54A24B", "#B279A2"],
    )
    fig.suptitle("Consistency overview: detection vs segmentation (orig vs multiviews)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(config.FIGURES_DIR / "consistency_overview.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_detection()
    plot_segmentation()
    plot_overview_panel()
    print(f"Saved figures to: {config.FIGURES_DIR}")
