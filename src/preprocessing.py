"""Steps 2-4 - CLEAN, MERGE, FEATURES."""
import numpy as np
import pandas as pd

# ONE source per variable (avoids duplicate/conflicting columns):
#  - temperature + cloud  -> solar_pv.csv (B's file, hourly, no gaps)
#  - humidity + wind      -> Darksky daily (not present in B's file)
DARKSKY_COLS = ["humidity", "windSpeed"]
WEATHER_COLS = ["temperatureMax", "temperatureMin", "cloudCover"] + DARKSKY_COLS
FEATURES = ["temperatureMax", "temperatureMin", "humidity", "windSpeed", "cloudCover",
            "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend", "is_holiday",
            "lag_1", "lag_7"]
TARGET = "kwh_per_household"

def clean_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Quality rules: drop duplicates, missing, negatives, and incomplete days
    (a full day has 48 half-hour readings; fewer => meter gap => biased sum)."""
    df = df.drop_duplicates(["LCLid", "day"]).dropna(subset=["energy_sum"])
    df = df[df["energy_sum"] >= 0]
    if "energy_count" in df:
        df = df[df["energy_count"] == 48]
    return df

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Households join/leave over time, so a raw total would be misleading.
    We use the MEAN consumption per household per day, plus the household count."""
    g = df.groupby("day").agg(kwh_per_household=("energy_sum", "mean"),
                              n_households=("LCLid", "nunique")).reset_index()
    return g[g["n_households"] >= 50].sort_values("day")   # drop the early sparse period

def solar_daily(solar: pd.DataFrame) -> pd.DataFrame:
    """Hourly (UTC) -> daily. Temperature/cloud become daily stats; PV is summed."""
    s = solar.copy(); s["day"] = s["timestamp"].dt.normalize()
    return s.groupby("day").agg(temperatureMax=("temp", "max"),
                                temperatureMin=("temp", "min"),
                                cloudCover=("cloud", "mean"),
                                irradiance_mean=("irradiance", "mean"),
                                pv_kwh=("pv_kwh", "sum")).reset_index()

def merge(daily, weather, holidays, solar) -> pd.DataFrame:
    out = daily.merge(solar_daily(solar), on="day", how="left")
    out = out.merge(weather[["day"] + DARKSKY_COLS], on="day", how="left")
    out["is_holiday"] = out["day"].isin(holidays["day"]).astype(int)
    out[WEATHER_COLS] = out[WEATHER_COLS].interpolate(limit_direction="both")
    return out

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("day").copy()
    dow, month = df["day"].dt.dayofweek, df["day"].dt.month
    # cyclic encoding: Sunday(6) is next to Monday(0), December next to January
    df["dow_sin"], df["dow_cos"] = np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)
    df["month_sin"], df["month_cos"] = np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12)
    df["is_weekend"] = (dow >= 5).astype(int)
    df["lag_1"] = df[TARGET].shift(1)   # yesterday
    df["lag_7"] = df[TARGET].shift(7)   # same weekday last week
    return df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

def time_split(df: pd.DataFrame, test_frac=0.2):
    """Chronological split. NEVER shuffle a time series (it leaks the future)."""
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut], df.iloc[cut:]

def build_dataset(daily, weather, holidays, solar) -> pd.DataFrame:
    return add_features(merge(aggregate(clean_daily(daily)), weather, holidays, solar))
