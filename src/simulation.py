"""Part C - SIMULATION: flat tariff vs dynamic agent, optionally with B's PV FORECAST.
Decision uses the FORECAST (what is known in advance); results are measured on the ACTUAL PV."""
import numpy as np
import pandas as pd
from .community import CommunityConfig, hourly_demand
from .pricing_agent import PricingAgent, TARIFFS, build_states, price_level

def respond(demand, price, shape, flex_share, sensitivity):
    """Demand response. Fixed part stays; the flexible part is SHIFTED (daily energy conserved)
    towards cheap hours, following the household's usual rhythm (shape)."""
    flex = flex_share * demand.sum()
    w = np.asarray(shape) * np.asarray(price) ** (-sensitivity)
    return (1 - flex_share) * demand + flex * w / w.sum()

def kpis(demand, pv, price) -> dict:
    net = demand - pv
    imp = np.clip(net, 0, None)
    return {"peak_grid_kw": float(imp.max()),
            "grid_import_kwh": float(imp.sum()),
            "pv_self_use_pct": float(100 * np.minimum(demand, pv).sum() / max(pv.sum(), 1e-9)),
            "peak_to_avg": float(demand.max() / demand.mean()),
            "cost_gbp": float((demand * price).sum() / 100)}

def simulate_day(kwh_per_household, is_weekend, pv_unit_h, agent: PricingAgent, profile,
                 cfg: CommunityConfig, pv_forecast_unit_h=None):
    """pv_unit_h: ACTUAL PV (24 values, one reference unit). pv_forecast_unit_h: B's forecast (optional).
    Summary keys: flat, dynamic, and 'oracle' (agent with perfect PV knowledge) when a forecast is given."""
    shape = profile["weekend" if is_weekend else "weekday"]
    base = hourly_demand(kwh_per_household, is_weekend, profile, cfg.n_households)
    scale = cfg.pv_units_per_household * cfg.n_households
    pv = np.asarray(pv_unit_h) * scale
    pv_seen = pv if pv_forecast_unit_h is None else np.asarray(pv_forecast_unit_h) * scale
    price_dyn = agent.predict(build_states(base, pv_seen, is_weekend))
    after = respond(base, price_dyn, shape, cfg.flex_share, cfg.sensitivity)
    # Revenue-neutral flat tariff: same average price the dynamic tariff charges the unshifted demand.
    # (Comparing with the fixed 'Normal' price would be unfair: dynamic prices average higher.)
    flat = np.full(24, np.average(price_dyn, weights=base))
    hourly = pd.DataFrame({"hour": range(24), "pv": pv, "pv_forecast": pv_seen, "demand_flat": base,
                           "demand_dynamic": after, "price": price_dyn, "level": price_level(price_dyn)})
    summary = {"flat": kpis(base, pv, flat), "dynamic": kpis(after, pv, price_dyn)}
    if pv_forecast_unit_h is not None:
        p_or = agent.predict(build_states(base, pv, is_weekend))
        summary["oracle"] = kpis(respond(base, p_or, shape, cfg.flex_share, cfg.sensitivity), pv, p_or)
    return hourly, summary

def simulate_period(df, predictions, pv_table, agent, profile, cfg, pv_forecast_table=None) -> pd.DataFrame:
    """Many days with Part A's PREDICTED consumption (+ B's forecast if given)."""
    rows = []
    for (_, r), pred in zip(df.iterrows(), predictions):
        d = r["day"]
        if d not in pv_table.index or (pv_forecast_table is not None and d not in pv_forecast_table.index):
            continue
        fc = None if pv_forecast_table is None else pv_forecast_table.loc[d].to_numpy()
        _, s = simulate_day(pred, bool(r["is_weekend"]), pv_table.loc[d].to_numpy(), agent, profile, cfg, fc)
        rows.append({"day": d, **{f"{k}_{m}": v for k, dd in s.items() for m, v in dd.items()}})
    out = pd.DataFrame(rows)
    out["peak_reduction_pct"] = 100 * (1 - out["dynamic_peak_grid_kw"] / out["flat_peak_grid_kw"])
    if "oracle_peak_grid_kw" in out:
        out["peak_reduction_oracle_pct"] = 100 * (1 - out["oracle_peak_grid_kw"] / out["flat_peak_grid_kw"])
    return out
