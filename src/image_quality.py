from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import lpips
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from . import config
from .utils import ImagePair, ensure_output_dirs, load_image_pairs


def _to_numpy(img) -> np.ndarray:
    return np.asarray(img).astype(np.float32) / 255.0


def _lpips_metric(model: lpips.LPIPS, pair: ImagePair) -> float:
    # LPIPS expects normalized torch tensors in NCHW format scaled to [-1, 1]
    def to_tensor(img):
        array = _to_numpy(img)
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return tensor * 2.0 - 1.0

    with torch.no_grad():
        score = model(to_tensor(pair.image_a), to_tensor(pair.image_b))
    return float(score.squeeze().cpu().item())


def compute_metrics(pairs: List[ImagePair]) -> pd.DataFrame:
    perceptual = lpips.LPIPS(net="alex")
    rows: List[Dict[str, float]] = []

    for pair in pairs:
        arr_a = _to_numpy(pair.image_a)
        arr_b = _to_numpy(pair.image_b)

        psnr = peak_signal_noise_ratio(arr_a, arr_b, data_range=1.0)
        ssim = structural_similarity(arr_a, arr_b, channel_axis=2, data_range=1.0)
        lpips_score = _lpips_metric(perceptual, pair)

        rows.append({
            "image": pair.stem,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips_score,
        })

    perceptual.cpu()
    return pd.DataFrame(rows).set_index("image")


def save_metrics_table(metrics: pd.DataFrame) -> Path:
    ensure_output_dirs()
    out_path = config.TABLES_DIR / "image_quality_metrics.csv"
    metrics.to_csv(out_path, float_format="{:.4f}".format)
    return out_path


def save_summary_table(metrics: pd.DataFrame) -> Path:
    summary = metrics.agg(["mean", "std"])
    formatted = summary.applymap(lambda x: f"{x:.4f}")
    summary_path = config.TABLES_DIR / "image_quality_summary.csv"
    formatted.to_csv(summary_path)
    return summary_path


def create_side_by_side_grid(pairs: List[ImagePair]) -> Path:
    ensure_output_dirs()

    cols = 2
    rows = math.ceil(len(pairs))
    fig, axes = plt.subplots(len(pairs), cols, figsize=(cols * 4, len(pairs) * 3))
    if len(pairs) == 1:
        axes = np.expand_dims(axes, axis=0)  # unify indexing

    for idx, pair in enumerate(pairs):
        axes[idx, 0].imshow(pair.image_a)
        axes[idx, 0].set_title(f"{pair.stem} - A")
        axes[idx, 1].imshow(pair.image_b)
        axes[idx, 1].set_title(f"{pair.stem} - B")
        for ax in axes[idx]:
            ax.axis("off")

    fig.tight_layout()
    out_path = config.FIGURES_DIR / "image_quality_side_by_side.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def run_image_quality_pipeline() -> Dict[str, Path]:
    pairs = load_image_pairs()
    metrics = compute_metrics(pairs)
    metrics_path = save_metrics_table(metrics)
    summary_path = save_summary_table(metrics)
    figure_path = create_side_by_side_grid(pairs)
    return {
        "metrics": metrics_path,
        "summary": summary_path,
        "figure": figure_path,
    }


if __name__ == "__main__":
    artifacts = run_image_quality_pipeline()
    for key, path in artifacts.items():
        print(f"{key}: {path}")
