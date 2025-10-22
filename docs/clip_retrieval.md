# CLIP 检索实验脚本说明

`src/clip_retrieval.py` 用于模拟下游检索任务：

1. 使用 CLIP ViT-B/32 提取 A 组（图库）与 B 组（查询）的特征。
2. 对每个查询计算与图库所有图片的余弦相似度，获取排名。
3. 统计 Top-1 是否找回同名图片，输出检索准确率。
4. 结果保存为：
   - `outputs/tables/clip_retrieval_summary.csv`：总体 Top-1 准确率。
   - `outputs/tables/clip_retrieval_details.csv`：每张查询图的检索排名与相似度，便于排查。

运行命令：

```powershell
python -m src.clip_retrieval
```

如需扩展到其他模型，可复用脚本结构，将 `_prepare_clip` 替换为目标模型的编解码逻辑即可。