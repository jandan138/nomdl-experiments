from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image

from . import config


@dataclass
class ImagePair:
    """Bundled PIL images plus their shared stem for saving artifacts."""

    stem: str
    image_a: Image.Image
    image_b: Image.Image


def ensure_output_dirs() -> None:
    """Create all output directories if they do not already exist."""
    for path in (config.OUTPUT_ROOT, config.FIGURES_DIR, config.TABLES_DIR, config.EMBEDDINGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_image_pairs(extensions: Iterable[str] = (".png", ".jpg", ".jpeg")) -> List[ImagePair]:
    """Return aligned image pairs from folders A/B.

    Images are matched by filename stem. Only files present in both folders are kept.
    """
    folder_a = config.DATA_ROOT / config.FOLDER_A
    folder_b = config.DATA_ROOT / config.FOLDER_B

    if not folder_a.exists() or not folder_b.exists():
        missing = [str(p) for p in (folder_a, folder_b) if not p.exists()]
        raise FileNotFoundError(f"Missing image folder(s): {', '.join(missing)}")

    def normalize(p: Path) -> Tuple[str, Path]:
        return p.stem, p

    allowed_exts = {ext.lower() for ext in extensions}

    images_a = {stem: path for stem, path in map(normalize, folder_a.iterdir()) if path.suffix.lower() in allowed_exts}
    images_b = {stem: path for stem, path in map(normalize, folder_b.iterdir()) if path.suffix.lower() in allowed_exts}

    shared_stems = sorted(set(images_a.keys()) & set(images_b.keys()))
    pairs: List[ImagePair] = []

    if shared_stems:
        for stem in shared_stems:
            img_a = Image.open(images_a[stem]).convert("RGB")
            img_b = Image.open(images_b[stem]).convert("RGB")
            pairs.append(ImagePair(stem=stem, image_a=img_a, image_b=img_b))
        return pairs

    files_a = sorted(images_a.items())
    files_b = sorted(images_b.items())

    if len(files_a) != len(files_b) or not files_a:
        raise ValueError("No aligned image pairs found. Check filenames and extensions.")

    print("[WARN] 未找到同名文件，将按排序顺序配对 A/B 图像。")
    for idx, ((stem_a, path_a), (stem_b, path_b)) in enumerate(zip(files_a, files_b)):
        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")
        stem = stem_b if stem_a != stem_b else stem_a
        if stem_a != stem_b:
            stem = f"pair_{idx:02d}_{stem_b}"
        pairs.append(ImagePair(stem=stem, image_a=img_a, image_b=img_b))
    return pairs
