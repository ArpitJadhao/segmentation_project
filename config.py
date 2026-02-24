# config.py
import os

# ── Paths ──────────────────────────────────────────────────────────────
DATASET_ROOT = "dataset"          # Change if dataset is elsewhere
TRAIN_RGB = os.path.join(DATASET_ROOT, "train", "Color_Images")
TRAIN_SEG = os.path.join(DATASET_ROOT, "train", "Segmentation")
VAL_RGB   = os.path.join(DATASET_ROOT, "val",   "Color_Images")
VAL_SEG   = os.path.join(DATASET_ROOT, "val",   "Segmentation")
TEST_RGB  = os.path.join(DATASET_ROOT, "testImages", "Color_Images")
TEST_SEG  = os.path.join(DATASET_ROOT, "testImages", "Segmentation")
RUNS_DIR     = "runs"

# ── Class Mapping ──────────────────────────────────────────────────────
ID2IDX = {
    100:   0,   # Trees
    200:   1,   # Lush Bushes
    300:   2,   # Dry Grass
    500:   3,   # Dry Bushes
    550:   4,   # Ground Clutter
    600:   5,   # Flowers
    700:   6,   # Logs
    800:   7,   # Rocks
    7100:  8,   # Landscape (default)
    10000: 9,   # Sky
}
IDX2NAME = {
    0: "Trees", 1: "Lush Bushes", 2: "Dry Grass",
    3: "Dry Bushes", 4: "Ground Clutter", 5: "Flowers",
    6: "Logs", 7: "Rocks", 8: "Landscape", 9: "Sky"
}
NUM_CLASSES = 10
DEFAULT_IDX = 8   # Unknown pixels → Landscape

# ── Visualization Colors (BGR for cv2 / RGB for matplotlib) ────────────
CLASS_COLORS = [
    (34,  139, 34),   # Trees        - Forest Green
    (0,   200, 0),    # Lush Bushes  - Lime Green
    (210, 180, 140),  # Dry Grass    - Tan
    (139, 90,  43),   # Dry Bushes   - Brown
    (128, 128, 128),  # Ground Clutter - Gray
    (255, 0,   255),  # Flowers      - Magenta
    (101, 67,  33),   # Logs         - Dark Brown
    (169, 169, 169),  # Rocks        - Dark Gray
    (194, 178, 128),  # Landscape    - Sand
    (135, 206, 235),  # Sky          - Sky Blue
]

# ── Training Hyperparameters ───────────────────────────────────────────
IMAGE_SIZE   = 640          # Resize both rgb and seg to this square size
BATCH_SIZE   = 4            # Reduce to 4 if GPU OOM
NUM_EPOCHS   = 80
LR           = 5e-5
WEIGHT_DECAY = 1e-4
PATIENCE     = 15           # Early stopping patience (epochs without val mIoU improvement)

# ── Model ──────────────────────────────────────────────────────────────
ENCODER = "efficientnet-b3"
ENCODER_WEIGHTS = "imagenet"
MODEL_NAME   = "DeepLabV3Plus"

# ── Misc ───────────────────────────────────────────────────────────────
SEED         = 42
NUM_WORKERS  = 4