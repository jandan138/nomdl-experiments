# 递归对比两个文件夹的成对图片

当两边目录结构一致（包含多级子目录）且相同相对路径下存在同名图片时，可使用 `src/compare_folders.py` 对所有图片进行批量对比。

## 适用目录示例
```
origin-figs/
  multi_views_with_bg_mdl/
    Component_0/
      view_00.png
    Component_1/
      view_00.png
  multi_views_with_bg_nomdl/
    Component_0/
      view_00.png
    Component_1/
      view_00.png
```

## 怎么运行
在项目根目录执行（Windows PowerShell）：

```powershell
# 仅需修改为你的两个根目录路径
python -m src.compare_folders `
  --folder-a "e:\my_dev\nomdl-experiments\origin-figs\multi_views_with_bg_mdl" `
  --folder-b "e:\my_dev\nomdl-experiments\origin-figs\multi_views_with_bg_nomdl" `
  --limit-figure 10
```

- `--folder-a`：A 侧根目录（参考图）。
- `--folder-b`：B 侧根目录（对比图）。
- `--limit-figure`：拼图中显示的成对图片数量（默认 10）。目录很大时建议只选取前若干张做可视化。

## 输出结果
- 指标表：`outputs/tables/image_quality_metrics.csv`（逐图 PSNR/SSIM/LPIPS）
- 汇总表：`outputs/tables/image_quality_summary.csv`（均值 ± 标准差）
- 对比拼图：`outputs/figures/image_quality_side_by_side.png`

表格字段详见 `docs/outputs_tables.md`。

## 匹配规则
- 按两侧的“相对路径”严格匹配，例如 `Component_1/view_00.png` 对 `Component_1/view_00.png`。
- 若 A 侧存在而 B 侧没有对应文件，会在终端打印警告并跳过该样本。

## 常见问题
- 图片过多导致 LPIPS 较慢：可先用 `--limit-figure` 控制拼图数量；指标计算会覆盖全部匹配样本。
- 图片格式支持：默认支持 `.png/.jpg/.jpeg`，需要可在 `utils.load_image_pairs_recursive` 中扩展。
