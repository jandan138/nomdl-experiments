from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torchvision import models

from . import config
from .utils import ImagePair, ensure_output_dirs, load_image_pairs, load_image_pairs_recursive


def _prepare_deeplab(device: torch.device):
    weights = models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model = models.segmentation.deeplabv3_resnet50(weights=weights)
    model.eval().to(device)
    preprocess = weights.transforms(resize_size=520)
    return model, preprocess, weights


def _predict_mask(model, preprocess, image, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        x = preprocess(image).unsqueeze(0).to(device)
        out = model(x)["out"]
        mask = out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)
    return mask


def _mean_iou_from_masks(a: np.ndarray, b: np.ndarray, ignore_index: int = 255) -> float:
    # compute mIoU over union of present classes, excluding ignore index
    classes = np.unique(np.concatenate([a.flatten(), b.flatten()]))
    classes = classes[classes != ignore_index]
    ious = []
    for c in classes:
        ta = (a == c)
        tb = (b == c)
        inter = np.logical_and(ta, tb).sum()
        union = np.logical_or(ta, tb).sum()
        if union == 0:
            continue
        ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def run_segmentation_consistency(pairs: Optional[List[ImagePair]] = None, tag: str = "orig") -> Dict[str, Path]:
    ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pairs is None:
        pairs = load_image_pairs()

    model, preprocess, weights = _prepare_deeplab(device)
    ignore_idx = weights.meta.get("ignore_background", None)
    if ignore_idx is None:
        ignore_idx = 255

    rows = []
    for pair in pairs:
        mask_a = _predict_mask(model, preprocess, pair.image_a, device)
        mask_b = _predict_mask(model, preprocess, pair.image_b, device)
        agree = float((mask_a == mask_b).mean())
        miou = _mean_iou_from_masks(mask_a, mask_b, ignore_index=ignore_idx)
        rows.append({"image": pair.stem, "pixel_agreement": agree, "miou_proxy": miou})

    df = pd.DataFrame(rows).set_index("image")
    out_csv = config.TABLES_DIR / f"segmentation_consistency_{tag}.csv"
    df.to_csv(out_csv, float_format="{:.4f}".format)

    summary = df.agg(["mean", "std"])  # no rounding to preserve
    out_sum = config.TABLES_DIR / f"segmentation_consistency_{tag}_summary.csv"
    summary.to_csv(out_sum)

    return {"detail": out_csv, "summary": out_sum}


def cli():
    ap = argparse.ArgumentParser(description="Segmentation consistency without labels (DeepLabV3)")
    ap.add_argument("--folder-a", type=Path, help="Root folder A (recursive)")
    ap.add_argument("--folder-b", type=Path, help="Root folder B (recursive)")
    ap.add_argument("--tag", type=str, default="orig", help="Suffix tag for outputs")
    args = ap.parse_args()

    if args.folder_a and args.folder_b:
        pairs = load_image_pairs_recursive(args.folder_a, args.folder_b)
    else:
        pairs = None

    arts = run_segmentation_consistency(pairs=pairs, tag=args.tag)
    for k, v in arts.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    cli()
