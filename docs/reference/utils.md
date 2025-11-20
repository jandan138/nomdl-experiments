# 工具函数说明

`src/utils.py` 提供加载图片与准备输出目录的基础功能：

- `ensure_output_dirs()`：自动创建结果输出所需的文件夹，无需手动准备。
- `load_image_pairs()`：优先按同名文件成对读取；若两侧文件名不同但数量一致，会给出提示并按排序顺序自动配对，依然返回 `ImagePair` 列表，方便快速实验。
- `load_image_pairs_recursive(folder_a, folder_b)`：从两个“根目录”开始递归遍历，按相对路径严格匹配成对图片，适合多级子目录批量对比（如 `Component_*/view_xx.png`）。
- `ImagePair` 数据类让单个图片对的处理更直观，包含 `stem`（文件名）以及两张 PIL 图像。

在新增实验脚本时只需调用这些工具函数，即可确保输入输出流程一致。
