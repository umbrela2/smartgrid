import numpy as np, pandas as pd
from src.preprocessing import clean_daily, build_dataset, time_split, FEATURES
from src.model_regression import train

def fake(n_days=400, n_hh=60):
    days = pd.date_range("2012-01-01", periods=n_days)
    rng = np.random.default_rng(0)
    temp = 10 + 8*np.sin(2*np.pi*np.arange(n_days)/365)
    rows = [(f"H{h}", d, 10 - 0.3*t + rng.normal(0, .5), 48) for h in range(n_hh)
            for d, t in zip(days, temp)]
    daily = pd.DataFrame(rows, columns=["LCLid", "day", "energy_sum", "energy_count"])
    w = pd.DataFrame({"day": days, "humidity": .7, "windSpeed": 3.})
    hrs = pd.date_range("2012-01-01", periods=n_days*24, freq="h")
    t = np.repeat(temp, 24)
    solar = pd.DataFrame({"timestamp": hrs, "irradiance": 100., "temp": t, "cloud": 50, "pv_kwh": .05})
    return daily, w, pd.DataFrame({"day": [days[5]], "holiday_type": ["x"]}), solar

def test_clean_drops_incomplete_days():
    d, _, _, _ = fake(); d.loc[0, "energy_count"] = 20
    assert len(clean_daily(d)) == len(d) - 1

def test_no_leakage_split_is_chronological():
    d, w, h, so = fake(); ds = build_dataset(d, w, h, so); tr, te = time_split(ds)
    assert tr["day"].max() < te["day"].min()

def test_model_beats_baseline():
    d, w, h, so = fake(); ds = build_dataset(d, w, h, so)
    _, m, b, _ = train(ds)
    assert m["MAE"] <= b["MAE"] and not ds[FEATURES].isna().any().any()
