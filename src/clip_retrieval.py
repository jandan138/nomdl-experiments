from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import open_clip
import pandas as pd
import torch
from torchvision import transforms

from . import config
from .utils import ImagePair, ensure_output_dirs, load_image_pairs


def _prepare_clip(device: torch.device):
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model.eval().to(device)

    def encode(images: torch.Tensor) -> torch.Tensor:
        return model.encode_image(images)

    return encode, preprocess


def _encode_pairs(pairs: List[ImagePair], device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    encode, preprocess = _prepare_clip(device)
    gallery_feats: List[np.ndarray] = []
    query_feats: List[np.ndarray] = []
    stems: List[str] = []

    with torch.no_grad():
        for pair in pairs:
            img_a = preprocess(pair.image_a).unsqueeze(0).to(device)
            img_b = preprocess(pair.image_b).unsqueeze(0).to(device)
            feat_a = encode(img_a)
            feat_b = encode(img_b)
            gallery_feats.append(feat_a.cpu().numpy().squeeze())
            query_feats.append(feat_b.cpu().numpy().squeeze())
            stems.append(pair.stem)

    return np.stack(gallery_feats), np.stack(query_feats), stems


def _normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, 1e-12, None)


def _evaluate_retrieval(gallery: np.ndarray, query: np.ndarray, stems: List[str]) -> pd.DataFrame:
    gallery = _normalize(gallery)
    query = _normalize(query)

    sims = query @ gallery.T
    ranks = np.argsort(-sims, axis=1)

    records = []
    correct = 0
    for i, stem in enumerate(stems):
        ordered_idx = ranks[i]
        top1 = ordered_idx[0]
        is_correct = stems[top1] == stem
        if is_correct:
            correct += 1
        match_rank = int(np.where(ordered_idx == i)[0][0]) + 1
        records.append({
            "query": stem,
            "top1_match": stems[top1],
            "rank": match_rank,
            "top1_score": float(sims[i, top1]),
        })

    accuracy = correct / len(stems)
    summary = pd.DataFrame([{"metric": "top1_accuracy", "value": accuracy}])
    detail = pd.DataFrame(records)
    return summary, detail


def run_clip_retrieval() -> Dict[str, Path]:
    ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = load_image_pairs()
    gallery, query, stems = _encode_pairs(pairs, device)

    summary, detail = _evaluate_retrieval(gallery, query, stems)

    summary_path = config.TABLES_DIR / "clip_retrieval_summary.csv"
    detail_path = config.TABLES_DIR / "clip_retrieval_details.csv"

    summary.to_csv(summary_path, index=False, float_format="{:.4f}".format)
    detail.to_csv(detail_path, index=False, float_format="{:.4f}".format)

    return {
        "summary": summary_path,
        "details": detail_path,
    }


if __name__ == "__main__":
    outputs = run_clip_retrieval()
    for name, path in outputs.items():
        print(f"{name}: {path}")
