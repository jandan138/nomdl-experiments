from __future__ import annotations

import argparse
from typing import Callable, Dict

from src.quality.image_quality import run_image_quality_pipeline
from src.representation.representation_analysis import run_representation_analysis
from src.representation.clip_retrieval import run_clip_retrieval


def _run_and_report(name: str, func: Callable[[], Dict[str, object]]) -> None:
    print(f"=== {name} ===")
    artifacts = func()
    for key, value in artifacts.items():
        print(f"{key}: {value}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="NomDL 实验流水线入口")
    parser.add_argument(
        "--stage",
        choices=["all", "quality", "representation", "retrieval"],
        default="all",
        help="选择要运行的实验阶段，默认全部执行",
    )
    args = parser.parse_args()

    if args.stage in ("all", "quality"):
        _run_and_report("Image Quality", run_image_quality_pipeline)
    if args.stage in ("all", "representation"):
        _run_and_report("Representation Analysis", run_representation_analysis)
    if args.stage in ("all", "retrieval"):
        _run_and_report("CLIP Retrieval", run_clip_retrieval)


if __name__ == "__main__":
    main()
