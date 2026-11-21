"""Shared C helpers: the simulated community + hourly data preparation."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class CommunityConfig:
    n_households: int = 100            # size of the simulated neighbourhood
    pv_units_per_household: float = 3.0  # multiples of B's reference PV system (ask B what 1 unit is)
    flex_share: float = 0.20           # share of consumption that can be shifted (EV, washing, heating...)
    sensitivity: float = 0.5           # how strongly flexible load reacts to price (0 = ignores price)

def pv_by_day(solar: pd.DataFrame) -> pd.DataFrame:
    """B's hourly file -> table: one row per day, columns 0..23 (kWh of one reference PV unit)."""
    s = solar.copy()
    s["day"] = s["timestamp"].dt.normalize(); s["hour"] = s["timestamp"].dt.hour
    t = s.pivot_table(index="day", columns="hour", values="pv_kwh", aggfunc="sum")
    return t.reindex(columns=range(24)).dropna()      # keep complete days only

def hourly_demand(kwh_per_household: float, is_weekend: bool, profile: dict, n_households: int) -> np.ndarray:
    """A predicts DAILY kWh per household; hourly = daily total x typical 24h shape."""
    shape = profile["weekend" if is_weekend else "weekday"]
    return kwh_per_household * n_households * np.asarray(shape)
