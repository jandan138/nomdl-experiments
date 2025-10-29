from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from . import config
from .utils import ImagePair, ensure_output_dirs, load_image_pairs, load_image_pairs_recursive


@dataclass
class Det:
    xyxy: np.ndarray  # [x1,y1,x2,y2]
    conf: float
    cls: int


def _load_yolo(device: torch.device):
    from ultralytics import YOLO
    # small model for speed; cached weights
    model = YOLO("yolov8n.pt")
    model.fuse()
    return model


def _predict(model, image) -> List[Det]:
    # model expects numpy array or path; we have PIL
    results = model.predict(image, verbose=False)
    dets: List[Det] = []
    for r in results:
        if r.boxes is None:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        for b, c, k in zip(boxes, confs, clss):
            dets.append(Det(b, float(c), int(k)))
    return dets


def _iou_matrix(a: List[Det], b: List[Det]) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    A = np.stack([d.xyxy for d in a])
    B = np.stack([d.xyxy for d in b])
    # IoU between all boxes
    # intersection
    tl = np.maximum(A[:, None, :2], B[None, :, :2])
    br = np.minimum(A[:, None, 2:], B[None, :, 2:])
    wh = np.clip(br - tl, a_min=0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    # areas
    area_a = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    area_b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.where(union > 0, inter / np.clip(union, 1e-9, None), 0.0)
    return iou.astype(np.float32)


def _greedy_match(a: List[Det], b: List[Det], iou_thr: float = 0.5, same_class: bool = True) -> Tuple[List[Tuple[int,int,float]], List[int], List[int]]:
    if not a or not b:
        return [], list(range(len(a))), list(range(len(b)))
    iou = _iou_matrix(a, b)
    # mask by class if required
    if same_class:
        cls_a = np.array([d.cls for d in a])[:, None]
        cls_b = np.array([d.cls for d in b])[None, :]
        mask = (cls_a == cls_b)
        iou = np.where(mask, iou, -1.0)

    matches: List[Tuple[int,int,float]] = []
    used_a = set()
    used_b = set()
    while True:
        i, j = np.unravel_index(np.argmax(iou), iou.shape)
        if iou[i, j] < iou_thr:
            break
        if i in used_a or j in used_b:
            iou[i, j] = -1.0
            continue
        matches.append((i, j, float(iou[i, j])))
        used_a.add(i)
        used_b.add(j)
        iou[i, :] = -1.0
        iou[:, j] = -1.0

    unmatched_a = [idx for idx in range(len(a)) if idx not in used_a]
    unmatched_b = [idx for idx in range(len(b)) if idx not in used_b]
    return matches, unmatched_a, unmatched_b


def _pair_metrics(dets_a: List[Det], dets_b: List[Det]) -> Dict[str, float]:
    matches, ua, ub = _greedy_match(dets_a, dets_b, iou_thr=0.5, same_class=True)
    count_a = len(dets_a)
    count_b = len(dets_b)
    matched = len(matches)
    mean_iou = float(np.mean([m[2] for m in matches])) if matches else 0.0
    coverage = matched / max(1, max(count_a, count_b))
    count_delta = abs(count_a - count_b)
    return {
        "count_a": float(count_a),
        "count_b": float(count_b),
        "matched": float(matched),
        "coverage": float(coverage),
        "mean_iou": float(mean_iou),
        "count_delta": float(count_delta),
    }


def run_detection_consistency(pairs: Optional[List[ImagePair]] = None, tag: str = "orig") -> Dict[str, Path]:
    ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pairs is None:
        pairs = load_image_pairs()

    model = _load_yolo(device)

    rows = []
    for pair in pairs:
        dets_a = _predict(model, pair.image_a)
        dets_b = _predict(model, pair.image_b)
        m = _pair_metrics(dets_a, dets_b)
        m.update({"image": pair.stem})
        rows.append(m)

    df = pd.DataFrame(rows).set_index("image")
    out_csv = config.TABLES_DIR / f"detection_consistency_{tag}.csv"
    df.to_csv(out_csv, float_format="{:.4f}".format)

    summary = df.agg(["mean", "std"]).round(4)
    out_sum = config.TABLES_DIR / f"detection_consistency_{tag}_summary.csv"
    summary.to_csv(out_sum)

    return {"detail": out_csv, "summary": out_sum}


def cli():
    ap = argparse.ArgumentParser(description="Detection consistency without labels (YOLOv8)")
    ap.add_argument("--folder-a", type=Path, help="Root folder A (recursive)")
    ap.add_argument("--folder-b", type=Path, help="Root folder B (recursive)")
    ap.add_argument("--tag", type=str, default="orig", help="Suffix tag for outputs")
    args = ap.parse_args()

    if args.folder_a and args.folder_b:
        pairs = load_image_pairs_recursive(args.folder_a, args.folder_b)
    else:
        pairs = None
    arts = run_detection_consistency(pairs=pairs, tag=args.tag)
    for k, v in arts.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    cli()
