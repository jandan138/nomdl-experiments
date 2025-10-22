from pathlib import Path

# Root directory containing the paired image folders
DATA_ROOT = Path("e:/my_dev/nomdl-experiments/origin-figs")

# Folder names that hold the paired images (A: high detail, B: simplified)
FOLDER_A = "mdl_images"
FOLDER_B = "nomdl_images"

# Output directories (created on demand)
OUTPUT_ROOT = Path("e:/my_dev/nomdl-experiments/outputs")
FIGURES_DIR = OUTPUT_ROOT / "figures"
TABLES_DIR = OUTPUT_ROOT / "tables"
EMBEDDINGS_DIR = OUTPUT_ROOT / "embeddings"

# Random seeds to keep visualizations stable
RANDOM_SEED = 42
