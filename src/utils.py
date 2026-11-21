"""Small helpers shared by the whole project (A, B and C import from here)."""
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"
SEED = 42

def regression_metrics(y_true, y_pred) -> dict:
    """MAE (same unit as target), RMSE (penalises big errors), R2 (1 = perfect)."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }
