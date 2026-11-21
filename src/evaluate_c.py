"""Final evaluation for the report:  python -m src.evaluate_c
Pipeline A (consumption) + B (PV forecast, test split only) -> C (prices) -> KPIs on ACTUAL PV."""
import numpy as np
import pandas as pd
from .community import CommunityConfig, pv_by_day
from .data_loader import load_daily, load_holidays, load_solar, load_weather
from .load_profile import get_profile
from .model_regression import predict_baseline_consumption
from .preprocessing import build_dataset, time_split
from .pricing_agent import PricingAgent
from .pv_forecast import forecast_tables, load_pv_predictions
from .simulation import simulate_period
from .utils import ROOT

def main():
    data = build_dataset(load_daily(), load_weather(), load_holidays(), load_solar())
    _, test_a = time_split(data)
    fc, act = forecast_tables(load_pv_predictions(), "test")
    days = test_a[test_a["day"].isin(fc.index)]
    profile, src = get_profile(); agent = PricingAgent.load()
    # sanity: B's y_true must equal the PV series we trained on
    ref = pv_by_day(load_solar()); common = act.index.intersection(ref.index)
    print(f"profile: {src} | evaluation days: {len(days)} | max |y_true - solar_pv|: "
          f"{float((act.loc[common] - ref.loc[common]).abs().to_numpy().max()):.2e}")
    preds = predict_baseline_consumption(days)
    out = []
    for units in (0, 3, 6, 10):
        cfg = CommunityConfig(pv_units_per_household=units)
        r = simulate_period(days, preds, act, agent, profile, cfg, fc)
        out.append({"PV units/household": units,
                    "peak cut % (B forecast)": r["peak_reduction_pct"].mean(),
                    "peak cut % (perfect PV)": r["peak_reduction_oracle_pct"].mean(),
                    "cost change %": 100 * (r["dynamic_cost_gbp"].sum() / r["flat_cost_gbp"].sum() - 1),
                    "PV self-use pts": (r["dynamic_pv_self_use_pct"] - r["flat_pv_self_use_pct"]).mean()})
    res = pd.DataFrame(out).round(2); print(res.to_string(index=False))
    res.to_csv(ROOT / "data" / "processed" / "sim_results.csv", index=False)

if __name__ == "__main__":
    main()
