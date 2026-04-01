"""Repository-root paths for data and trained artifacts."""

from pathlib import Path

# src/imugesture/paths.py -> parents[2] == repository root
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_CSV = REPO_ROOT / "data" / "gesture_data.csv"
MODEL_JOBLIB = REPO_ROOT / "models" / "model.joblib"


def ensure_data_dirs() -> None:
    """Ensure ``data/`` and ``models/`` exist."""
    (REPO_ROOT / "data").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "models").mkdir(parents=True, exist_ok=True)
