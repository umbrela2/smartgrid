"""Tests for the Part A CSV export (synthetic data: the raw Kaggle files are not needed)."""
import subprocess, sys
from pathlib import Path
import joblib
import numpy as np, pandas as pd
import pytest
from src.part_a.export_predictions import COLUMNS, DEFAULT_MODEL, build_prediction_table, export_predictions
from src.part_a.model_regression import save, train
from src.part_a.preprocessing import FEATURES, TARGET, build_dataset

ROOT = Path(__file__).resolve().parents[1]

def fake(n_days=400, n_hh=60):
    days = pd.date_range("2012-01-01", periods=n_days)
    rng = np.random.default_rng(0)
    temp = 10 + 8*np.sin(2*np.pi*np.arange(n_days)/365)
    rows = [(f"H{h}", d, 10 - 0.3*t + rng.normal(0, .5), 48) for h in range(n_hh) for d, t in zip(days, temp)]
    daily = pd.DataFrame(rows, columns=["LCLid", "day", "energy_sum", "energy_count"])
    w = pd.DataFrame({"day": days, "humidity": .7, "windSpeed": 3.})
    hrs = pd.date_range("2012-01-01", periods=n_days*24, freq="h")
    solar = pd.DataFrame({"timestamp": hrs, "irradiance": 100., "temp": np.repeat(temp, 24), "cloud": 50, "pv_kwh": .05})
    return daily, w, pd.DataFrame({"day": [days[5]], "holiday_type": ["x"]}), solar

@pytest.fixture(scope="module")
def dataset():
    return build_dataset(*fake())

@pytest.fixture
def model_path(dataset, tmp_path):
    model, *_ = train(dataset); p = tmp_path / "m.joblib"; save(model, p); return p

def test_columns_and_row_count(dataset, model_path):
    t = build_prediction_table(dataset, model_path)
    assert list(t.columns) == COLUMNS and len(t) == len(dataset)
    assert not t.isna().any().any() and t["day"].is_unique and t["day"].is_monotonic_increasing

def test_predictions_come_from_the_saved_model(dataset, model_path):
    t = build_prediction_table(dataset, model_path)
    expected = joblib.load(model_path).predict(dataset.sort_values("day")[FEATURES])
    np.testing.assert_allclose(t["predicted_kwh_per_household"], expected)
    np.testing.assert_allclose(t["actual_kwh_per_household"], dataset.sort_values("day")[TARGET])
    np.testing.assert_allclose(t["error_kwh_per_household"], t["predicted_kwh_per_household"] - t["actual_kwh_per_household"])

def test_split_is_chronological_80_20(dataset, model_path):
    t = build_prediction_table(dataset, model_path)
    assert (t["split"].iloc[:-1].tolist() == sorted(t["split"].iloc[:-1], reverse=True))  # train... then test...
    assert t.loc[t.split == "train", "day"].max() < t.loc[t.split == "test", "day"].min()
    assert len(t[t.split == "test"]) == len(dataset) - int(len(dataset) * 0.8)

def test_write_read_roundtrip_and_split_filter(dataset, model_path, tmp_path):
    out = export_predictions(dataset, tmp_path / "sub" / "pred.csv", model_path, split="test")
    r = pd.read_csv(out, parse_dates=["timestamp"])
    assert out.exists() and set(r["split"]) == {"test"} and list(r.columns) == COLUMNS
    assert (r["timestamp"].dt.hour == 0).all() and r["day"].str.len().eq(10).all()
    with pytest.raises(ValueError): export_predictions(dataset, tmp_path / "x.csv", model_path, split="bad")

def test_missing_columns_and_model(dataset, tmp_path):
    with pytest.raises(ValueError): build_prediction_table(dataset.drop(columns=["lag_7"]), DEFAULT_MODEL)
    with pytest.raises(FileNotFoundError): build_prediction_table(dataset, tmp_path / "nope.joblib")

def test_shipped_model_is_compatible_with_export(dataset):
    """The preserved models/consumption_regression.joblib must work with the export unchanged."""
    t = build_prediction_table(dataset, DEFAULT_MODEL)
    assert len(t) == len(dataset) and np.isfinite(t["predicted_kwh_per_household"]).all()

def _modules_after_import(stmt):
    code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); {stmt}; "
            "print(' '.join(m for m in sys.modules if m.startswith('src.')))")
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return set(r.stdout.split())

def test_part_a_does_not_import_part_c():
    mods = _modules_after_import("import src.part_a.export_predictions")
    assert not any(m.startswith("src.part_c") for m in mods)
