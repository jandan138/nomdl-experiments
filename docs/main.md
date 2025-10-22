# 全流程入口脚本

`src/main.py` 将三个实验阶段统一到一个命令行接口中，便于一键复现或单独运行。

用法示例：

```powershell
# 依次执行图像质量、表征分析、CLIP 检索
python -m src.main

# 只运行图像质量实验
python -m src.main --stage quality

# 只做表征分析
python -m src.main --stage representation

# 只做 CLIP 检索
python -m src.main --stage retrieval
```

脚本会在控制台打印每个阶段输出的文件路径，方便快速定位实验结果。