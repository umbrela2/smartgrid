"""Part A - export the consumption predictions to CSV.

    python -m src.part_a.export_predictions                 (all days  -> data/processed/pred_consumption.csv)
    python -m src.part_a.export_predictions --split test    (held-out days only)

What the CSV holds (one row = one DAY, mean consumption of one household):
    timestamp                     start of the day, 'YYYY-MM-DD 00:00:00' (same text format as Part B's CSV)
    day                           'YYYY-MM-DD' (join key used by Part C)
    predicted_kwh_per_household   output of the saved Part A model (models/consumption_regression.joblib)
    actual_kwh_per_household      measured value from the smart-meter data
    error_kwh_per_household       predicted - actual
    split                         'train' or 'test' = the same chronological 80/20 split used to train/evaluate A
    n_households                  number of meters averaged that day
    is_weekend, is_holiday        calendar flags (0/1) used by the model / needed by C

The predictions come from `predict_baseline_consumption` (the real Part A prediction step).
Nothing is simulated or hard-coded. This module does not import Part B or Part C.

Prediction mode: the model uses lag_1 / lag_7 (the MEASURED consumption of the previous day / week),
so each row is a one-day-ahead prediction made with real history, not a multi-day recursive forecast.
"""
import argparse
from pathlib import Path

import pandas as pd

from ..utils import MODELS, ROOT
from .model_regression import predict_baseline_consumption
from .preprocessing import FEATURES, TARGET, time_split

DEFAULT_MODEL = MODELS / "consumption_regression.joblib"
DEFAULT_OUT = ROOT / "data" / "processed" / "pred_consumption.csv"
COLUMNS = ["timestamp", "day", "predicted_kwh_per_household", "actual_kwh_per_household",
           "error_kwh_per_household", "split", "n_households", "is_weekend", "is_holiday"]


def build_prediction_table(dataset: pd.DataFrame, model_path=DEFAULT_MODEL, test_frac: float = 0.2) -> pd.DataFrame:
    """dataset = output of preprocessing.build_dataset(...). Returns the export table (not yet written)."""
    missing = [c for c in FEATURES + [TARGET, "day", "n_households"] if c not in dataset.columns]
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Part A model not found: {model_path}. Train it with: python -m src.part_a.model_regression")
    ds = dataset.sort_values("day").reset_index(drop=True)
    train_df, _ = time_split(ds, test_frac)               # same split as model_regression.train()
    pred = predict_baseline_consumption(ds, path=model_path)
    out = pd.DataFrame({
        "timestamp": ds["day"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "day": ds["day"].dt.strftime("%Y-%m-%d"),
        "predicted_kwh_per_household": pred,
        "actual_kwh_per_household": ds[TARGET].to_numpy(),
        "split": ["train" if i < len(train_df) else "test" for i in range(len(ds))],
        "n_households": ds["n_households"].astype(int).to_numpy(),
        "is_weekend": ds["is_weekend"].astype(int).to_numpy(),
        "is_holiday": ds["is_holiday"].astype(int).to_numpy(),
    })
    out["error_kwh_per_household"] = out["predicted_kwh_per_household"] - out["actual_kwh_per_household"]
    if out["predicted_kwh_per_household"].isna().any():
        raise ValueError("model produced NaN predictions")
    return out[COLUMNS]


def export_predictions(dataset: pd.DataFrame, out_path=DEFAULT_OUT, model_path=DEFAULT_MODEL,
                       split: str = "all") -> Path:
    """Predict with the saved Part A model and write the CSV. split: 'all' | 'train' | 'test'."""
    if split not in ("all", "train", "test"):
        raise ValueError("split must be 'all', 'train' or 'test'")
    table = build_prediction_table(dataset, model_path)
    if split != "all":
        table = table[table["split"] == split]
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export Part A consumption predictions to CSV.")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output CSV (default: data/processed/pred_consumption.csv)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="Part A model (default: models/consumption_regression.joblib)")
    ap.add_argument("--split", choices=["all", "train", "test"], default="all", help="rows to export (default: all)")
    a = ap.parse_args(argv)
    from .data_loader import load_daily, load_holidays, load_solar, load_weather
    from .preprocessing import build_dataset
    try:
        data = build_dataset(load_daily(), load_weather(), load_holidays(), load_solar())
    except FileNotFoundError as e:
        raise SystemExit(f"Missing input file: {e.filename}\nPart A needs data/raw/daily_dataset.csv, "
                         "weather_daily_darksky.csv, uk_bank_holidays.csv (Kaggle 'smart-meters-in-london') "
                         "and data/processed/solar_pv.csv.")
    path = export_predictions(data, a.out, a.model, a.split)
    t = pd.read_csv(path)
    print(f"Wrote {len(t)} rows to {path}  (days {t['day'].min()} -> {t['day'].max()}, split={a.split})")


if __name__ == "__main__":
    main()
