"""Step 1 - LOAD raw Kaggle files (jeanmidev/smart-meters-in-london).
Put these in data/raw/: daily_dataset.csv, weather_daily_darksky.csv, uk_bank_holidays.csv
"""
import pandas as pd
from .utils import RAW, ROOT

def load_daily(path=RAW / "daily_dataset.csv") -> pd.DataFrame:
    # one row = one household (LCLid) on one day
    return pd.read_csv(path, parse_dates=["day"])

def load_weather(path=RAW / "weather_daily_darksky.csv") -> pd.DataFrame:
    w = pd.read_csv(path)
    w["day"] = pd.to_datetime(w["time"]).dt.normalize()
    return w

def load_holidays(path=RAW / "uk_bank_holidays.csv") -> pd.DataFrame:
    h = pd.read_csv(path)
    h.columns = ["day", "holiday_type"]
    h["day"] = pd.to_datetime(h["day"])
    return h

def load_solar(path=ROOT / "data" / "processed" / "solar_pv.csv") -> pd.DataFrame:
    """From B: hourly UTC file with irradiance, temp, cloud, pv_kwh."""
    return pd.read_csv(path, parse_dates=["timestamp"])
