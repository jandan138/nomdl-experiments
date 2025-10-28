from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from . import config
from .image_quality import compute_metrics, create_side_by_side_grid, save_metrics_table, save_summary_table
from .utils import ensure_output_dirs, load_image_pairs_recursive


def run_compare_folders(folder_a: Path, folder_b: Path, limit_figure: int | None = 10) -> Dict[str, Path]:
    ensure_output_dirs()
    pairs = load_image_pairs_recursive(folder_a, folder_b)

    metrics = compute_metrics(pairs)
    metrics_path = save_metrics_table(metrics)
    summary_path = save_summary_table(metrics)

    if limit_figure is not None:
        vis_pairs = pairs[:limit_figure]
    else:
        vis_pairs = pairs
    figure_path = create_side_by_side_grid(vis_pairs)
    return {
        "metrics": metrics_path,
        "summary": summary_path,
        "figure": figure_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two folder trees with paired images at same relative paths")
    parser.add_argument("--folder-a", required=True, type=Path, help="Reference folder (A)")
    parser.add_argument("--folder-b", required=True, type=Path, help="Comparison folder (B)")
    parser.add_argument("--limit-figure", type=int, default=10, help="Number of pairs to visualize side-by-side (default 10)")
    args = parser.parse_args()

    artifacts = run_compare_folders(args.folder_a, args.folder_b, args.limit_figure)
    for k, v in artifacts.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
