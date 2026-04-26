"""Project configuration."""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

SAMPLE_DATA_PATH = DATA_DIR / "sample_data.csv"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
FEATURE_CACHE_PATH = ARTIFACTS_DIR / "fused_features.npy"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
GEMMA_MODEL_NAME = "google/gemma-2-2b-it"

RANDOM_SEED = 42
TEST_SIZE = 0.25

LABEL_ORDER = ["Low", "Medium", "High"]

GEMMA_SCORE_KEYS = [
    "hook_strength",
    "emotional_appeal",
    "clarity",
    "curiosity_gap",
    "cta_strength",
]
