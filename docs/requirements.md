# 环境依赖安装说明

项目根目录提供 `requirements.txt`，包含运行上述实验所需的第三方库（Torch、OpenCLIP、LPIPS 等）。

建议使用虚拟环境，依次执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

若已具备 GPU 环境，可根据 PyTorch 官网指引替换为匹配 CUDA 版本的安装命令，然后再执行其余依赖的安装。