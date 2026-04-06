from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
RUNS_ROOT = ARTIFACTS_ROOT / "runs"
PROCESSED_ROOT = ARTIFACTS_ROOT / "processed_hits"
DATASET_META_FILENAME = "dataset.json"

SOURCE_SECTION_NAMES = ("pooled", "validation")
SOURCE_SECTION_LABELS = {
    "pooled": "Train/Test Pool",
    "validation": "Validation",
}
SOURCE_SECTION_ALIASES = {
    "pooled": "pooled",
    "training": "pooled",
    "testing": "pooled",
    "validation": "validation",
}

SECTION_NAMES = ("training", "testing", "validation")
SECTION_LABELS = {
    "training": "Training",
    "testing": "Testing",
    "validation": "Validation",
}

DEFAULT_DATASET_SLUG = "workspace"
DEFAULT_DATASET_LABEL = "Workspace"
EXAMPLE_DATASET_SLUG = "example-hw3"
EXAMPLE_DATASET_LABEL = "Example - HW3"

ALLOWED_AUDIO_EXTENSIONS = {
    ".wav",
    ".m4a",
    ".mp3",
    ".flac",
    ".ogg",
    ".webm",
}

DEFAULT_RECORDING_PREFIX = "S"
DEFAULT_GOOD_HITS = 10
DEFAULT_BAD_HITS = 24
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_DELTA_LIST = [0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
