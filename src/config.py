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

# --- FVC Datasets Configuration ---
FVC_BASE_DIR = DATA_DIR / "fvc200X"
FVC_DATASET_NAMES = ["FVC2000", "FVC2002", "FVC2004"]
FVC_DBS = ["Db1_", "Db2_", "Db3_", "Db4_"]

# --- AgeDB Specific Paths ---
AGEDB_ROOT = DATA_DIR / "AgeDB"

# --- CelebA Specific Paths ---
CELEBA_ROOT = DATA_DIR / "CelebA"
CELEBA_ATTR_FILE = CELEBA_ROOT / "list_attr_celeba.csv"
CELEBA_PARTITION_FILE = CELEBA_ROOT / "list_eval_partition.csv"
CELEBA_IMAGE_DIR = CELEBA_ROOT / "img_align_celeba" / "img_align_celeba"
CELEBA_SENSITIVE_ATTRIBUTES = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]
