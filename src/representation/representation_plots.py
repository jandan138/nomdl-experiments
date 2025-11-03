from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.core import config


def _load_embeddings(path: Path):
    data = np.load(path, allow_pickle=True)
    stems = [s for s in data["stems"]]
    feats_a = data["feats_a"]
    feats_b = data["feats_b"]
    return stems, feats_a, feats_b


def _cosines_from_feats(feats_a: np.ndarray, feats_b: np.ndarray) -> np.ndarray:
    a = feats_a / np.linalg.norm(feats_a, axis=1, keepdims=True)
    b = feats_b / np.linalg.norm(feats_b, axis=1, keepdims=True)
    return np.sum(a * b, axis=1)


def violin_cosine_plot(emb_paths: Dict[str, Path], out_path: Path) -> Path:
    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 1, figsize=(8, 5))

    labels = []
    data = []
    for label, p in emb_paths.items():
        if not p.exists():
            continue
        _, feats_a, feats_b = _load_embeddings(p)
        cos = _cosines_from_feats(feats_a, feats_b)
        labels.append(label)
        data.append(cos)

    parts = axes.violinplot(data, showmeans=False, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#1f77b4")
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)

    axes.set_xticks(np.arange(1, len(labels) + 1))
    axes.set_xticklabels(labels, rotation=30)
    axes.set_ylabel("Pairwise cosine similarity")
    axes.set_ylim(-0.1, 1.0)
    axes.set_title("Per-pair cosine similarity distributions")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def bar_metrics_plot(summary_paths: Dict[str, Path], out_path: Path) -> Path:
    metrics = ["cosine_mean", "fid", "sliced_wasserstein"]
    summaries: Dict[str, pd.DataFrame] = {}
    for ds_label, p in summary_paths.items():
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df = df.set_index("model")
        summaries[ds_label] = df

    models = set()
    for df in summaries.values():
        models.update(df.index.tolist())
    models = sorted(models)
    datasets = sorted(summaries.keys())

    n_metrics = len(metrics)
    x = np.arange(len(models))
    width = 0.7 / max(1, len(datasets))

    plt.style.use("ggplot")
    fig, axes = plt.subplots(nrows=1, ncols=n_metrics, figsize=(4 * n_metrics, 4), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for j, ds in enumerate(datasets):
            df = summaries.get(ds)
            vals = [df.loc[m, metric] if (df is not None and m in df.index and pd.notna(df.loc[m, metric])) else np.nan for m in models]
            pos = x - (len(datasets) - 1) * width / 2 + j * width
            ax.bar(pos, vals, width=width, label=ds, color=colors[j % len(colors)], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30)
        ax.set_title(metric)
        if metric == "cosine_mean":
            ax.set_ylim(0, 1.0)
    axes[0].legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> Dict[str, Path]:
    ensure_dirs = [config.FIGURES_DIR]
    for d in ensure_dirs:
        d.mkdir(parents=True, exist_ok=True)

    emb_candidates = {
        "CLIP": config.EMBEDDINGS_DIR / "clip_features.npz",
        "DINOv2": config.EMBEDDINGS_DIR / "dinov2_features.npz",
        "CLIP_multiviews": config.EMBEDDINGS_DIR / "clip_features_multiviews.npz",
        "DINOv2_multiviews": config.EMBEDDINGS_DIR / "dinov2_features_multiviews.npz",
    }

    summary_candidates = {
        "orig": config.TABLES_DIR / "representation_summary.csv",
        "multiviews": config.TABLES_DIR / "representation_summary_multiviews.csv",
    }

    violin_path = config.FIGURES_DIR / "representation_cosine_violins.png"
    bar_path = config.FIGURES_DIR / "representation_metrics_bars.png"

    violin_out = violin_cosine_plot(emb_candidates, violin_path)
    bar_out = bar_metrics_plot(summary_candidates, bar_path)

    return {"violins": violin_out, "bars": bar_out}


if __name__ == "__main__":
    out = main()
    for k, v in out.items():
        print(f"{k}: {v}")
