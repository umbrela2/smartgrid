"""B's PV forecasts (data/processed/pred_pv.csv: timestamp, y_pred, y_true, split).
RULE: evaluate the pricing agent on split == 'test' only (train/val were seen by B's LSTM)."""
import pandas as pd
from .utils import ROOT

PRED_CSV = ROOT / "data" / "processed" / "pred_pv.csv"

def load_pv_predictions(path=PRED_CSV) -> pd.DataFrame:
    p = pd.read_csv(path, parse_dates=["timestamp"])
    p["y_pred"] = p["y_pred"].clip(lower=0)       # PV cannot be negative
    return p

def _table(p, col):
    p = p.assign(day=p["timestamp"].dt.normalize(), hour=p["timestamp"].dt.hour)
    return p.pivot_table(index="day", columns="hour", values=col).reindex(columns=range(24)).dropna()

def forecast_tables(preds: pd.DataFrame, split: str = "test"):
    """Returns (forecast, actual): day x 24 tables, COMPLETE days only (B's file has gaps at split edges)."""
    p = preds[preds["split"] == split]
    f, a = _table(p, "y_pred"), _table(p, "y_true")
    days = f.index.intersection(a.index)
    return f.loc[days], a.loc[days]
