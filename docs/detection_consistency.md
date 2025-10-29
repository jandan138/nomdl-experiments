# 检测一致性（无标注代理）

目的：不依赖标注数据，衡量 A/B 配对图在“目标检测任务”上的一致性，近似评估材质简化对下游检测的影响。

方法：使用预训练 YOLOv8n 对 A、B 各自独立推理，按 IoU>0.5 且同类别进行贪心匹配，统计：
- count_a / count_b：各侧检测框数量
- matched：成功匹配的数量
- coverage：matched / max(count_a, count_b)（越高越一致）
- mean_iou：匹配框的平均 IoU（越高越一致）
- count_delta：|count_a - count_b|（越小越一致）

运行命令（默认比较 `mdl_images` vs `nomdl_images` 并输出 `*_orig.csv`）：
```powershell
conda activate nomdl
python -m src.detection_consistency --tag orig
```

递归比较多级目录（例如 multiviews），并输出 `*_multiviews.csv`：
```powershell
python -m src.detection_consistency `
  --folder-a "e:\my_dev\nomdl-experiments\origin-figs\multi_views_with_bg_mdl" `
  --folder-b "e:\my_dev\nomdl-experiments\origin-figs\multi_views_with_bg_nomdl" `
  --tag multiviews
```

输出：
- `outputs/tables/detection_consistency_{tag}.csv`：逐图指标
- `outputs/tables/detection_consistency_{tag}_summary.csv`：均值/标准差汇总
