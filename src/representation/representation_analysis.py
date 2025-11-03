from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
import open_clip
import pandas as pd
import torch
from scipy import linalg
from sklearn.manifold import TSNE
from torchvision import transforms
import timm
from timm.data import resolve_data_config, create_transform

from src.core import config
from src.core.utils import ImagePair, ensure_output_dirs, load_image_pairs


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


@dataclass
class FeatureBatch:
    stems: List[str]
    feats_a: np.ndarray
    feats_b: np.ndarray


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(a_norm * b_norm, axis=1)


def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        eps = 1e-6
        sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
        sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(fid)


def _sliced_wasserstein(a: np.ndarray, b: np.ndarray, num_projections: int = 512) -> float:
    dim = a.shape[1]
    projections = np.random.randn(num_projections, dim)
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)
    distances = []
    for proj in projections:
        proj_a = np.dot(a, proj)
        proj_b = np.dot(b, proj)
        proj_a.sort()
        proj_b.sort()
        distances.append(np.mean(np.abs(proj_a - proj_b)))
    return float(np.mean(distances))


def _prepare_clip(device: torch.device) -> Tuple[Callable[[torch.Tensor], torch.Tensor], transforms.Compose]:
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model.eval().to(device)

    def forward(images: torch.Tensor) -> torch.Tensor:
        return model.encode_image(images)

    return forward, preprocess


def _prepare_dino(device: torch.device) -> Tuple[Callable[[torch.Tensor], torch.Tensor], transforms.Compose]:
    """Prepare DINOv2 encoder. Prefer official torch.hub; fallback to timm weights.

    Returns a (forward, preprocess) pair.
    """
    try:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        model.eval().to(device)

        def forward(images: torch.Tensor) -> torch.Tensor:
            return model(images)

        preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        return forward, preprocess
    except Exception as e:
        print(f"[INFO] 官方 DINOv2 hub 加载失败，尝试使用 timm 预训练权重。原因：{e}")

    model = timm.create_model(
        "vit_small_patch14_dinov2",
        pretrained=True,
        num_classes=0,
        global_pool="token",
    ).to(device)
    model.eval()

    cfg = resolve_data_config({}, model=model)
    preprocess = create_transform(**cfg)

    def forward(images: torch.Tensor) -> torch.Tensor:
        return model(images)

    return forward, preprocess


def _extract_features(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    preprocess,
    pairs: List[ImagePair],
    device: torch.device,
) -> FeatureBatch:
    stems: List[str] = []
    feats_a: List[np.ndarray] = []
    feats_b: List[np.ndarray] = []

    with torch.no_grad():
        for pair in pairs:
            for img, collector in ((pair.image_a, feats_a), (pair.image_b, feats_b)):
                tensor = preprocess(img).unsqueeze(0).to(device)
                outputs = forward_fn(tensor)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                collector.append(outputs.cpu().numpy().squeeze())
            stems.append(pair.stem)

    return FeatureBatch(stems=stems, feats_a=np.stack(feats_a), feats_b=np.stack(feats_b))


def _normalize_features(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, a_min=1e-12, a_max=None)


def _save_embeddings(name: str, batch: FeatureBatch, suffix: str = "") -> Path:
    ensure_output_dirs()
    suf = f"_{suffix}" if suffix else ""
    path = config.EMBEDDINGS_DIR / f"{name}_features{suf}.npz"
    np.savez_compressed(path, stems=batch.stems, feats_a=batch.feats_a, feats_b=batch.feats_b)
    return path


def _tsne_plot(clip_batch: FeatureBatch, suffix: str = "") -> Path:
    ensure_output_dirs()
    combined = np.concatenate([clip_batch.feats_a, clip_batch.feats_b], axis=0)
    labels = ["A"] * len(clip_batch.feats_a) + ["B"] * len(clip_batch.feats_b)
    tsne = TSNE(n_components=2, perplexity=5, random_state=config.RANDOM_SEED)
    embedding = tsne.fit_transform(combined)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))
    for domain, color in ("A", "tab:blue"), ("B", "tab:orange"):
        mask = [lbl == domain for lbl in labels]
        plt.scatter(embedding[mask, 0], embedding[mask, 1], label=domain, alpha=0.7)
    plt.legend()
    plt.title("CLIP Feature t-SNE")
    plt.tight_layout()

    suf = f"_{suffix}" if suffix else ""
    out_path = config.FIGURES_DIR / f"representation_tsne{suf}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def run_representation_analysis(pairs: Optional[List[ImagePair]] = None, suffix: str = "") -> Dict[str, Path]:
    ensure_output_dirs()
    _set_seed(config.RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pairs is None:
        pairs = load_image_pairs()

    clip_forward, clip_preprocess = _prepare_clip(device)
    clip_batch = _extract_features(clip_forward, clip_preprocess, pairs, device)
    clip_embeddings_path = _save_embeddings("clip", clip_batch, suffix)

    artifacts = {
        "clip_embeddings": clip_embeddings_path,
    }

    dino_batch = None
    try:
        dino_forward, dino_preprocess = _prepare_dino(device)
        dino_batch = _extract_features(dino_forward, dino_preprocess, pairs, device)
        dino_embeddings_path = _save_embeddings("dinov2", dino_batch, suffix)
        artifacts["dinov2_embeddings"] = dino_embeddings_path
    except Exception as exc:
        print(f"[WARN] DINOv2 加载失败（{exc}），将跳过相关指标。")

    summary_rows = []

    for name, batch in (("CLIP", clip_batch), ("DINOv2", dino_batch)):
        if batch is None:
            continue
        cosines = _cosine_similarity(batch.feats_a, batch.feats_b)
        feats_a = _normalize_features(batch.feats_a)
        feats_b = _normalize_features(batch.feats_b)

        mean_a = feats_a.mean(axis=0)
        mean_b = feats_b.mean(axis=0)
        cov_a = np.cov(feats_a, rowvar=False)
        cov_b = np.cov(feats_b, rowvar=False)

        fid = _frechet_distance(mean_a, cov_a, mean_b, cov_b)
        swd = _sliced_wasserstein(feats_a, feats_b)

        summary_rows.append({
            "model": name,
            "cosine_mean": float(cosines.mean()),
            "cosine_std": float(cosines.std()),
            "fid": float(fid),
            "sliced_wasserstein": float(swd),
        })

    suf = f"_{suffix}" if suffix else ""
    summary_path = config.TABLES_DIR / f"representation_summary{suf}.csv"
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(summary_path, index=False, float_format="{:.4f}".format)
    else:
        summary_path.write_text("model,cosine_mean,cosine_std,fid,sliced_wasserstein\n")

    json_path = config.TABLES_DIR / f"representation_summary{suf}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    tsne_path = _tsne_plot(clip_batch, suffix)

    artifacts.update({
        "summary_csv": summary_path,
        "summary_json": json_path,
        "tsne": tsne_path,
    })
    return artifacts


if __name__ == "__main__":
    paths = run_representation_analysis()
    for label, path in paths.items():
        print(f"{label}: {path}")
