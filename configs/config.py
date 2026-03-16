"""VectorWorld project constants and path resolution."""
from pathlib import Path
import os

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
_DEFAULT_CFG_DIR = PROJECT_ROOT / "configs"
CONFIG_PATH = Path(os.getenv("CONFIG_PATH", _DEFAULT_CFG_DIR)).as_posix()

# Dataset constants
NUM_WAYMO_TRAIN_SCENARIOS = 487002

# Scene layout type
NON_PARTITIONED = 0
PARTITIONED = 1

# Partition mask
AFTER_PARTITION = 0
BEFORE_PARTITION = 1

# Waymo lane connection types
LANE_CONNECTION_TYPES_WAYMO = {
    "none": 0, "pred": 1, "succ": 2, "left": 3, "right": 4, "self": 5
}

# NuPlan lane connection types
LANE_CONNECTION_TYPES_NUPLAN = {
    "none": 0, "pred": 1, "succ": 2, "self": 3
}

# Nocturne compatibility
PROPORTION_NOCTURNE_COMPATIBLE = 0.38
NOCTURNE_COMPATIBLE = 1

# NuPlan agent types
NUPLAN_VEHICLE = 0
NUPLAN_PEDESTRIAN = 1
NUPLAN_STATIC_OBJECT = 2

# Unified agent state format: [pos_x, pos_y, speed, cos_h, sin_h, length, width]
UNIFIED_FORMAT_INDICES = {
    "pos_x": 0, "pos_y": 1, "speed": 2,
    "cos_heading": 3, "sin_heading": 4,
    "length": 5, "width": 6,
}