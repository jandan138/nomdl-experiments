# NomDL Experiments

该仓库包含对两组配对图像（原材质 vs 简化材质）的三类评估实验：

1. **图像质量实验**：计算 PSNR / SSIM / LPIPS，并输出 10 对图片的左右对比拼图。
2. **表征层分析**：使用 CLIP / DINOv2 提取特征，统计余弦相似度、FID、Sliced Wasserstein，并生成 t-SNE 可视化。
3. **下游检索模拟**：基于 CLIP 评估 Top-1 检索准确率，观测任务层性能变化。

> 📁 配套说明文档位于 `docs/`，输出制品保存在 `outputs/`。

---

## 目录结构

```text
├─ origin-figs/               # 原始配对图像（A 组 vs B 组）
│   ├─ mdl_images/            # A 组：原材质图片
│   └─ nomdl_images/          # B 组：简化材质图片
├─ src/                       # 核心脚本
│   ├─ config.py              # 全局路径、随机种子配置
│   ├─ utils.py               # 加载图片对、输出目录工具
│   ├─ image_quality.py       # 图像质量实验
│   ├─ representation_analysis.py # 表征层分析
│   ├─ clip_retrieval.py      # CLIP 检索实验
│   └─ main.py                # 命令行入口，串联全部实验
├─ docs/                      # 新手友好操作指南
│   └─ ...                    # 每个脚本对应一份说明
├─ outputs/                   # 实验结果（脚本运行后自动生成）
│   ├─ figures/               # 拼图、t-SNE 等可视化
│   ├─ tables/                # 指标表格
│   └─ embeddings/            # 中间特征缓存
└─ requirements.txt           # Python 依赖列表
```

---

## 快速开始

### 1. 克隆仓库
```bash
git clone <YOUR-REPO-URL>
cd nomdl-experiments
```

### 2. 准备 Conda 环境
```bash
conda create --name nomdl python=3.10 -y
conda activate nomdl
```

> 若遇到 `SSL` 下载错误，可改用清华镜像：
> ```bash
> conda create --name nomdl python=3.10 -y \
>   --override-channels \
>   -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
>   -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
> ```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

> Windows 环境若提示 `SSL: UNEXPECTED_EOF_WHILE_READING`，多尝试几次或配置代理即可成功。

### 4. 放置图片
确保 `origin-figs/mdl_images` 与 `origin-figs/nomdl_images` 各包含同数量图片。若文件名不同（如 `orbit_mdl_00.png` vs `orbit_00.png`），脚本会自动按排序顺序配对，但建议使用一致命名以便追踪结果。

### 5. 运行全部实验
```bash
python -m src.main
```

- 终端会依次打印各阶段的输出路径。
- 结果保存到 `outputs/`，首次运行会自动创建目录。

如需单独运行某一阶段：
```bash
python -m src.main --stage quality          # 仅图像质量
python -m src.main --stage representation   # 仅表征层分析
python -m src.main --stage retrieval        # 仅 CLIP 检索
```

---

## 输出文件速览
- 指标表格：`outputs/tables/`（详见 `docs/outputs_tables.md`）。
- 关键图像：`outputs/figures/image_quality_side_by_side.png`、`outputs/figures/representation_tsne.png`。
- 特征缓存：`outputs/embeddings/*.npz`，可复用或做更多分析。

---

## 常见问题

- **DINOv2 权重下载失败**：网络无法访问 GitHub 时会跳过 DINO 指标，终端提示 `[WARN] DINOv2 加载失败`。配置代理或提前下载权重后，再运行 `python -m src.representation_analysis` 可补齐结果。
- **Hugging Face 下载慢**：安装 `huggingface_hub[hf_xet]` 可提高读取速度，或设置镜像/代理。
- **LPIPS 第一次运行很慢**：需下载 AlexNet 权重，一次成功后会缓存。

---

## 贡献指南
1. Fork & Pull Request：欢迎在本仓库基础上扩展模型或新增指标。
2. 提交前：请运行 `python -m src.main` 确认关键脚本通过。
3. 代码风格：保持 `src/` 中已有的结构与注释风格，必要时更新 `docs/` 对应指导文档。

如有问题欢迎提交 Issue！
