# 图像质量实验脚本使用指南

`src/image_quality.py` 自动完成图像质量对比实验：

1. 读取 `mdl_images` 与 `nomdl_images` 中同名的图片对。
2. 计算每对图片的 PSNR、SSIM 与 LPIPS 指标，并输出到 `outputs/tables/image_quality_metrics.csv`。
3. 生成均值±标准差汇总表 `outputs/tables/image_quality_summary.csv`，方便直接引用。
4. 创建左右对比拼图 `outputs/figures/image_quality_side_by_side.png`，展示 10 对图像的视觉差别。

运行方法（建议先在环境中安装 `lpips`, `scikit-image`, `matplotlib`, `pandas` 等依赖）：

```powershell
python -m src.image_quality
```

脚本会自动创建缺失的输出目录，并在控制台打印生成的文件路径。