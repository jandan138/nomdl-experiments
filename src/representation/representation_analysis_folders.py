from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from src.core.utils import ensure_output_dirs, load_image_pairs_recursive
from src.representation.representation_analysis import run_representation_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Representation analysis on two folder trees with same relative paths")
    parser.add_argument("--folder-a", required=True, type=Path, help="Root folder A (reference)")
    parser.add_argument("--folder-b", required=True, type=Path, help="Root folder B (comparison)")
    parser.add_argument("--tag", type=str, default="multiviews", help="Suffix tag for output filenames to avoid overwrite")
    args = parser.parse_args()

    pairs = load_image_pairs_recursive(args.folder_a, args.folder_b)
    artifacts = run_representation_analysis(pairs=pairs, suffix=args.tag)
    for k, v in artifacts.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
