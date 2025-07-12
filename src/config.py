"""
Configurations for datasets and output directories.
"""

from pathlib import Path

# --- Global Paths ---
ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "datasets"
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- LFW Specific Paths ---
LFW_ROOT = DATA_DIR / "lfw"
LFW_PAIRS_FILE = LFW_ROOT / "pairs.csv"
LFW_IMAGE_DIR = LFW_ROOT / "lfw-deepfunneled"

# --- CASIA-Iris-Thousand Specific Paths ---
CASIA_ROOT = DATA_DIR / "casia-iris-thousand/CASIA-Iris-Thousand"
