# 分割一致性（无标注代理）

目的：不依赖标注数据，衡量 A/B 配对图在“语义分割任务”上的一致性，近似评估材质简化对分割的影响。

方法：使用 torchvision 预训练 DeepLabV3-ResNet50（COCO with VOC labels）对 A、B 独立推理：
- pixel_agreement：像素级一致率（A/B 预测类别相同像素的占比）。
- miou_proxy：将两侧预测视作“伪标签”，在类的并集中计算 mean IoU。

运行命令（默认比较 `mdl_images` vs `nomdl_images` 并输出 `*_orig.csv`）：
```powershell
conda activate nomdl
python -m src.segmentation_consistency --tag orig
```

递归比较多级目录（例如 multiviews），并输出 `*_multiviews.csv`：
```powershell
python -m src.segmentation_consistency `
  --folder-a "e:\my_dev\nomdl-experiments\origin-figs\multi_views_with_bg_mdl" `
  --folder-b "e:\my_dev\nomdl-experiments\origin-figs\multi_views_with_bg_nomdl" `
  --tag multiviews
```

输出：
- `outputs/tables/segmentation_consistency_{tag}.csv`：逐图指标
- `outputs/tables/segmentation_consistency_{tag}_summary.csv`：均值/标准差汇总

提示：首次运行会下载模型权重，建议保持网络可用或配置镜像。
