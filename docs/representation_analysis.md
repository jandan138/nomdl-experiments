# 表征层分析脚本说明

`src/representation_analysis.py` 主要用于比较 CLIP 与 DINOv2 对两组图片的特征差异。

运行步骤概览：

1. 载入图片对并复现随机种子，确保可重复性。
2. 用 CLIP ViT-B/32 与 DINOv2 ViT-S/14 提取特征，保存到 `outputs/embeddings/*.npz`，便于后续复用。
3. 对每个模型计算：
   - 成对余弦相似度的均值与标准差（影响越小越接近 1）。
   - FID（Frechet Inception Distance）衡量整体分布差异。
   - 切片 Wasserstein 距离（Sliced Wasserstein Distance）观察分布偏移。
4. 将结果导出到 `outputs/tables/representation_summary.csv` 与 `representation_summary.json`。
5. 用 CLIP 特征做 t-SNE 可视化，生成 `outputs/figures/representation_tsne.png`。

依赖建议：`torch`, `open_clip_torch`, `torchvision`, `scipy`, `scikit-learn`, `matplotlib`。

运行命令：

```powershell
python -m src.representation_analysis
```

脚本会自动检测 GPU（若可用）以加速推理；若缺少某些模型或依赖，可按照错误提示安装。