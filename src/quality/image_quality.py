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

from src.core import config
from src.core.utils import ImagePair, ensure_output_dirs, load_image_pairs


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
    """Summarize metrics with principled handling of ±inf values.

    Policy:
    - Convert to numeric; detect ±inf in PSNR.
    - Replace ±inf PSNR with a finite "reasonable" value instead of dropping:
        replacement = min(percentile_99_9(finite_psnr), finite_psnr.max()) if available, else 60.0
      This approximates an upper-but-realistic bound and avoids skew from true infinity.
    - Aggregate mean/std over the imputed series.
    - Write a side meta CSV with inf count and the replacement used.
    """
    numeric = metrics.apply(pd.to_numeric, errors="coerce")

    # Determine replacement for inf PSNR
    psnr_series = numeric.get("psnr", pd.Series(dtype=float))
    inf_mask = np.isinf(psnr_series.values) if psnr_series is not None else np.array([], dtype=bool)
    inf_count = int(inf_mask.sum())
    total = int(len(numeric))

    replacement = None
    if psnr_series is not None and len(psnr_series) > 0:
        finite_psnr = psnr_series.replace([np.inf, -np.inf], np.nan).dropna().values
        if finite_psnr.size > 0:
            # 99.9th percentile (robust to rare extremes); fallback to max if needed
            try:
                p99_9 = float(np.percentile(finite_psnr, 99.9))
            except Exception:
                p99_9 = float(np.nanmax(finite_psnr))
            replacement = float(min(p99_9, float(np.nanmax(finite_psnr))))
        else:
            replacement = 60.0
    else:
        replacement = 60.0

    # Impute infs in PSNR with the chosen replacement
    if psnr_series is not None and inf_count > 0:
        numeric.loc[inf_mask, "psnr"] = replacement

    # Aggregate summary
    agg_df = numeric.agg(["mean", "std"]).round(4)
    summary_path = config.TABLES_DIR / "image_quality_summary.csv"
    agg_df.to_csv(summary_path)

    # Persist meta information
    meta = pd.DataFrame({
        "psnr_inf_count": [inf_count],
        "psnr_inf_replacement": [replacement],
        "n_pairs": [total],
    })
    meta_path = config.TABLES_DIR / "image_quality_summary_meta.csv"
    meta.to_csv(meta_path, index=False)

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
