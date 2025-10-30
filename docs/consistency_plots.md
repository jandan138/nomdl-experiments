# Consistency plots

This script visualizes the new label-free consistency metrics for detection and segmentation, comparing the original 10 pairs (orig) vs the multiviews dataset.

Figures generated to `outputs/figures`:

- `consistency_detection_bars.png`: mean coverage and mean IoU of matched detections
- `consistency_detection_violin.png`: distributions of coverage and mean IoU per image
- `consistency_segmentation_bars.png`: mean pixel agreement and mIoU proxy
- `consistency_segmentation_violin.png`: distributions of pixel agreement and mIoU proxy per image
- `consistency_overview.png`: a compact 2x2 panel for quick storytelling

## How to run

The script reads tables from `outputs/tables` produced by `src/detection_consistency.py` and `src/segmentation_consistency.py`.

```bash
python -m src.consistency_plots
```

On Windows PowerShell, ensure your conda environment is active:

```powershell
conda activate nomdl
python -m src.consistency_plots
```

## Storyline

- Detection is sensitive to view/background changes: coverage and IoU drop notably from `orig` to `multiviews`.
- Segmentation stays comparatively stable: high pixel agreement and similar mIoU proxy across datasets.
- Taken together, this suggests that while object-level detection fluctuates with viewpoint/background, the overall scene segmentation remains consistent—supporting the hypothesis that A/B differences minimally affect global semantics but can disrupt instance-level detection.
