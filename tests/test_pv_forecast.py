import numpy as np, pandas as pd
from src.pv_forecast import forecast_tables

def test_only_complete_test_days_and_nonnegative():
    ts = pd.date_range("2013-11-01 05:00", periods=24 + 24 + 10, freq="h")   # partial day at both ends
    p = pd.DataFrame({"timestamp": ts, "y_pred": 0.1, "y_true": 0.1, "split": "test"})
    p = pd.concat([p, p.assign(split="train", timestamp=ts - pd.Timedelta(days=300))])
    f, a = forecast_tables(p, "test")
    assert len(f) == 1 and f.shape[1] == 24 and (f.index == a.index).all()
