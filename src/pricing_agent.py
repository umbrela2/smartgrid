"""Part C core - the dense-network PRICING AGENT.

State (per hour)  : hour of day, weekend flag, demand and PV relative to the day's mean demand.
Action (output)   : price in pence/kWh, between the Low and High tariff of the London trial.
How it learns     : a 'scarcity' rule (teacher) gives the target price from the net load
                    (demand - PV). The neural network learns this policy from real days.
                    Idea: expensive when the grid is stressed, cheap when PV is in surplus.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .community import CommunityConfig, hourly_demand
from .utils import MODELS, SEED

# UK Power Networks dynamic-ToU trial levels (p/kWh) - VERIFY against your dataset's tariff file.
TARIFFS = {"Low": 3.99, "Normal": 11.76, "High": 67.20}
P_LOW, P_HIGH = TARIFFS["Low"], TARIFFS["High"]
STATE_COLS = ["hour_sin", "hour_cos", "is_weekend", "demand_rel", "pv_rel", "net_rel"]
PV_LEVELS = (0, 1, 3, 6, 10)   # train on several PV penetrations so the agent generalises

def build_states(demand_h, pv_h, is_weekend) -> pd.DataFrame:
    """24 rows (one per hour). Everything is relative to mean hourly demand => scale-free."""
    m = np.mean(demand_h); h = np.arange(24)
    d, p = np.asarray(demand_h) / m, np.asarray(pv_h) / m
    return pd.DataFrame({"hour_sin": np.sin(2*np.pi*h/24), "hour_cos": np.cos(2*np.pi*h/24),
                         "is_weekend": float(is_weekend), "demand_rel": d, "pv_rel": p, "net_rel": d - p})

def teacher_price(net_rel, center=1.1, steep=6.0):
    """Smooth scarcity rule on a log scale: net load << avg -> Low, >> avg -> High."""
    s = 1 / (1 + np.exp(-steep * (np.asarray(net_rel) - center)))
    return np.exp(np.log(P_LOW) + s * (np.log(P_HIGH) - np.log(P_LOW)))

def price_level(p):
    p = np.asarray(p)
    return np.where(p < 6.5, "Low", np.where(p > 30, "High", "Normal"))

def build_training_set(df, pv_table, profile, n_households=100, pv_levels=PV_LEVELS):
    """df: Part A dataset (needs day, kwh_per_household, is_weekend). Uses REAL daily consumption
    and REAL hourly PV from B."""
    parts = []
    for _, row in df.iterrows():
        if row["day"] not in pv_table.index: continue
        dem = hourly_demand(row["kwh_per_household"], bool(row["is_weekend"]), profile, n_households)
        for lvl in pv_levels:
            pv = pv_table.loc[row["day"]].to_numpy() * lvl * n_households
            parts.append(build_states(dem, pv, row["is_weekend"]))
    X = pd.concat(parts, ignore_index=True)
    return X, teacher_price(X["net_rel"])

class PricingAgent:
    def __init__(self):
        self.model = make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(32, 16), activation="relu", max_iter=800,
            early_stopping=True, random_state=SEED))

    def fit(self, X, prices):
        self.model.fit(X[STATE_COLS], np.log(prices)); return self   # learn log-price

    def predict(self, X) -> np.ndarray:
        return np.clip(np.exp(self.model.predict(X[STATE_COLS])), P_LOW, P_HIGH)

    def save(self, path=MODELS / "pricing_agent.joblib"):
        MODELS.mkdir(exist_ok=True); joblib.dump(self.model, path)

    @classmethod
    def load(cls, path=MODELS / "pricing_agent.joblib"):
        a = cls(); a.model = joblib.load(path); return a

if __name__ == "__main__":
    from .data_loader import load_daily, load_weather, load_holidays, load_solar
    from .preprocessing import build_dataset, time_split
    from .community import pv_by_day
    from .load_profile import get_profile
    data = build_dataset(load_daily(), load_weather(), load_holidays(), load_solar())
    train_df, test_df = time_split(data)
    prof, src = get_profile(); pv = pv_by_day(load_solar())
    Xtr, ytr = build_training_set(train_df, pv, prof); Xte, yte = build_training_set(test_df, pv, prof)
    agent = PricingAgent().fit(Xtr, ytr)
    pred = agent.predict(Xte)
    print(f"profile: {src} | train {len(Xtr)} / test {len(Xte)} states")
    print("MAE (p/kWh):", round(float(np.mean(np.abs(pred - yte))), 3))
    print("Level accuracy:", round(float(np.mean(price_level(pred) == price_level(yte))), 3))
    agent.save()
