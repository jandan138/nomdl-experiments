# 表征实验的可视化（彩色曲线 / 小提琴图与指标柱状图）

本说明介绍如何生成并解读两张彩色对比图：

1. representation_cosine_violins.png：每对图片的余弦相似度分布（小提琴图），比较 CLIP 与 DINOv2（包含 multiviews 后缀若存在）。
2. representation_metrics_bars.png：柱状图展示每个模型在不同数据集上的指标（cosine_mean、fid、sliced_wasserstein），便于横向对比。

如何生成

```powershell
conda activate nomdl
python -m src.representation_plots
```

当脚本发现 `outputs/embeddings/*.npz` 与 `outputs/tables/representation_summary*.csv` 时会自动使用它们生成图；如果没有会跳过对应图例。

如何解读

- 小提琴图（violin）：每个小提琴代表一组（例如 `CLIP`、`DINOv2`、`CLIP_multiviews`），宽度反映密度，图中竖直位置为余弦相似度（-0.1 到 1.0）。越靠近 1 表示多数配对在特征上非常相似；DINOv2 曲线若更扁平或偏低，说明它对材质变化更敏感。

- 指标柱状图（bars）：每个模块显示不同数据集（orig / multiviews）下的指标值，方便看哪种模型在整体分布（FID）和配对相似度（cosine_mean）上表现更稳定。

如果你希望：
- 把 violin 改为 boxplot 或 density 曲线，我可以替换显示风格；
- 增加 per-component 聚合（按子文件夹统计），我可以再加一个脚本产出按 Component 分组的折线图。
