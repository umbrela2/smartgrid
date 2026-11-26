"""Self-contained tests for the loaders and preprocessing steps (tiny synthetic files, no Kaggle data)."""
import numpy as np
import pandas as pd
import pytest

from src.part_a.data_loader import load_daily, load_holidays, load_solar, load_weather
from src.part_a.preprocessing import (DARKSKY_COLS, FEATURES, TARGET, add_features, aggregate,
                                      build_dataset, clean_daily, merge, solar_daily, time_split)


def test_loaders_read_expected_file_formats(tmp_path):
    pd.DataFrame({"LCLid": ["A", "B"], "day": ["2012-01-01", "2012-01-01"], "energy_sum": [1.0, 2.0],
                  "energy_count": [48, 48]}).to_csv(tmp_path / "d.csv", index=False)
    pd.DataFrame({"time": ["2012-01-01 00:00:00", "2012-01-02 00:00:00"], "humidity": [.5, .6],
                  "windSpeed": [1., 2.]}).to_csv(tmp_path / "w.csv", index=False)
    pd.DataFrame({"Bank holidays": ["2012-01-02"], "Type": ["New Year"]}).to_csv(tmp_path / "h.csv", index=False)
    pd.DataFrame({"timestamp": ["2012-01-01 00:00:00"], "irradiance": [0.], "temp": [5.], "cloud": [50],
                  "pv_kwh": [0.]}).to_csv(tmp_path / "s.csv", index=False)
    assert pd.api.types.is_datetime64_any_dtype(load_daily(tmp_path / "d.csv")["day"])
    w = load_weather(tmp_path / "w.csv")
    assert w["day"].tolist() == list(pd.to_datetime(["2012-01-01", "2012-01-02"]))
    h = load_holidays(tmp_path / "h.csv")
    assert list(h.columns) == ["day", "holiday_type"] and h["day"].iloc[0] == pd.Timestamp("2012-01-02")
    assert pd.api.types.is_datetime64_any_dtype(load_solar(tmp_path / "s.csv")["timestamp"])


def test_clean_daily_rules():
    df = pd.DataFrame({"LCLid": ["A", "A", "B", "C", "D"],
                       "day": pd.to_datetime(["2012-01-01"] * 2 + ["2012-01-01"] * 3),
                       "energy_sum": [1.0, 1.0, -1.0, np.nan, 3.0],
                       "energy_count": [48, 48, 48, 48, 47]})
    out = clean_daily(df)
    assert out["LCLid"].tolist() == ["A"]   # duplicate, negative, NaN and incomplete (47) rows removed


def test_aggregate_mean_per_household_and_min_50_households():
    rows = [(f"H{i}", pd.Timestamp("2012-01-01"), float(i), 48) for i in range(60)]
    rows += [(f"H{i}", pd.Timestamp("2012-01-02"), 1.0, 48) for i in range(10)]   # sparse day -> dropped
    g = aggregate(pd.DataFrame(rows, columns=["LCLid", "day", "energy_sum", "energy_count"]))
    assert len(g) == 1 and g["n_households"].iloc[0] == 60
    assert g[TARGET].iloc[0] == pytest.approx(np.mean(range(60)))


def test_solar_daily_hourly_to_daily():
    ts = pd.date_range("2012-01-01", periods=48, freq="h")
    s = pd.DataFrame({"timestamp": ts, "irradiance": 10., "temp": np.r_[np.arange(24), np.arange(24) + 100],
                      "cloud": 40, "pv_kwh": 1.})
    d = solar_daily(s)
    assert len(d) == 2 and d["temperatureMax"].tolist() == [23, 123] and d["temperatureMin"].tolist() == [0, 100]
    assert d["pv_kwh"].tolist() == [24., 24.]


def test_merge_flags_holidays_and_has_no_missing_weather():
    days = pd.date_range("2012-01-01", periods=5)
    daily = pd.DataFrame({"day": days, TARGET: 1., "n_households": 60})
    weather = pd.DataFrame({"day": days, "humidity": .5, "windSpeed": 2.})
    hol = pd.DataFrame({"day": [days[2]], "holiday_type": ["x"]})
    solar = pd.DataFrame({"timestamp": pd.date_range("2012-01-01", periods=120, freq="h"),
                          "irradiance": 1., "temp": 7., "cloud": 30, "pv_kwh": 0.})
    m = merge(daily, weather, hol, solar)
    assert m["is_holiday"].tolist() == [0, 0, 1, 0, 0]
    assert not m[["temperatureMax", "temperatureMin", "cloudCover"] + DARKSKY_COLS].isna().any().any()


def test_add_features_lags_and_calendar():
    days = pd.date_range("2012-01-02", periods=20)   # 2012-01-02 is a Monday
    df = pd.DataFrame({"day": days, TARGET: np.arange(20, dtype=float), "n_households": 60, "is_holiday": 0,
                       "temperatureMax": 1., "temperatureMin": 0., "cloudCover": 1., "humidity": 1., "windSpeed": 1.})
    f = add_features(df)
    assert len(f) == 13 and f["day"].iloc[0] == days[7]
    assert (f["lag_1"] == f[TARGET] - 1).all() and (f["lag_7"] == f[TARGET] - 7).all()
    assert f["is_weekend"].tolist() == [int(d.dayofweek >= 5) for d in f["day"]]
    assert set(FEATURES) <= set(f.columns)


def test_time_split_is_80_20_and_chronological():
    df = pd.DataFrame({"day": pd.date_range("2012-01-01", periods=100)})
    tr, te = time_split(df)
    assert (len(tr), len(te)) == (80, 20) and tr["day"].max() < te["day"].min()


def test_build_dataset_end_to_end_with_synthetic_inputs():
    from tests.test_pipeline import fake
    ds = build_dataset(*fake())
    assert {"day", TARGET, "n_households", "is_weekend", "is_holiday"} <= set(ds.columns)
    assert not ds[FEATURES + [TARGET]].isna().any().any()
