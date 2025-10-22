# `outputs/tables` 结果解读指南

运行 `python -m src.main` 后，所有实验的表格型结果都会汇总到 `outputs/tables/` 目录。以下逐一说明每个文件的来源、内容与阅读方式。

## 1. `image_quality_metrics.csv`
- **来源**：`src/image_quality.py`。
- **内容**：逐对图片的图像质量指标，列包括：
  - `image`：图片对的名称（默认取 B 组文件名，或自动生成的 `pair_xx_xxx`）。
  - `psnr`：峰值信噪比，数值越高代表差异越小。
  - `ssim`：结构相似度，范围在 0~1，越接近 1 越相似。
  - `lpips`：感知差异指标，越接近 0 表示人眼上越难区分。
- **用途**：观察每对图片的具体差异，识别是否存在明显偏离的样本。

## 2. `image_quality_summary.csv`
- **来源**：`src/image_quality.py`。
- **内容**：对 `image_quality_metrics.csv` 的聚合统计，提供 `mean`（平均值）与 `std`（标准差）。
- **用途**：一眼了解整体视觉差异水平，可直接引用到报告或 PPT 中。

## 3. `representation_summary.csv`
- **来源**：`src/representation_analysis.py`。
- **内容**：针对特征表征实验的汇总，每行对应一个模型（目前成功运行的模型为 CLIP，DINOv2 下载失败时会跳过）。列包括：
  - `model`：模型名称，如 `CLIP`。
  - `cosine_mean` / `cosine_std`：成对图片的余弦相似度均值与标准差。
  - `fid`：Frechet 距离，衡量整体分布差异，越低越好。
  - `sliced_wasserstein`：切片 Wasserstein 距离，同样越低越接近。
- **用途**：快速比较不同视觉模型对材质差异的敏感程度。

## 4. `representation_summary.json`
- **来源**：`src/representation_analysis.py`。
- **内容**：与 CSV 相同的信息，以 JSON 格式保存，便于被其他脚本或前端直接读取。
- **用途**：适合需要机器可读的场景，如自动生成报告或仪表盘。

## 5. `clip_retrieval_summary.csv`
- **来源**：`src/clip_retrieval.py`。
- **内容**：CLIP 检索实验的总体 Top-1 准确率：
  - `metric`: 固定为 `top1_accuracy`。
  - `value`: 准确率数值，范围 0~1。
- **用途**：评估简化材质是否影响检索任务表现，可视作任务层一致性的量化指标。

## 6. `clip_retrieval_details.csv`
- **来源**：`src/clip_retrieval.py`。
- **内容**：逐个查询图像的检索结果：
  - `query`：查询图像名称（B 组）。
  - `top1_match`：检索返回的图库图像（A 组）名称。
  - `rank`：正确匹配在排行榜中的名次（1 表示成功命中 Top-1）。
  - `top1_score`：Top-1 匹配的余弦相似度。越接近 1 越相似。
- **用途**：定位检索失败的样本、分析相似度分布，可视化或进一步调试检索策略。

---

> 小贴士：若目录中缺少某个文件，说明对应实验尚未运行或中途失败。可使用 `python -m src.main --stage ...` 重新跑对应阶段。