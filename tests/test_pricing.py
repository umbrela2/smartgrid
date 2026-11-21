import numpy as np
from src.community import CommunityConfig
from src.load_profile import get_profile
from src.pricing_agent import P_HIGH, P_LOW, PricingAgent, build_states, teacher_price, price_level
from src.simulation import respond, simulate_day

PROF, _ = get_profile()
H = np.arange(24)
PV_UNIT = np.clip(np.sin((H - 6) / 12 * np.pi), 0, None) * 0.3   # fake sunny day

def trained_agent():
    xs, ys = [], []
    for k in (0.5, 1.0, 1.5):
        for wk in (False, True):
            dem = 100 * 10 * np.asarray(PROF["weekend" if wk else "weekday"]) * k
            for lvl in (0, 3, 8):
                st = build_states(dem, PV_UNIT * lvl * 100, wk); xs.append(st); ys.append(teacher_price(st["net_rel"]))
    import pandas as pd
    return PricingAgent().fit(pd.concat(xs, ignore_index=True), np.concatenate(ys))

def test_teacher_bounds_and_monotonic():
    p = teacher_price(np.linspace(-1, 3, 50))
    assert p.min() >= P_LOW - 1e-9 and p.max() <= P_HIGH + 1e-9 and np.all(np.diff(p) >= 0)

def test_demand_response_conserves_energy():
    d = 10 * np.asarray(PROF["weekday"]); p = np.linspace(4, 60, 24)
    out = respond(d, p, PROF["weekday"], 0.3, 0.6)
    assert abs(out.sum() - d.sum()) < 1e-9 and out[np.argmin(p)] > d[np.argmin(p)]

def test_agent_learns_and_reduces_peak():
    agent = trained_agent()
    cfg = CommunityConfig()
    hourly, s = simulate_day(10.0, False, PV_UNIT, agent, PROF, cfg)
    assert hourly["price"].between(P_LOW, P_HIGH).all()
    assert s["dynamic"]["peak_grid_kw"] <= s["flat"]["peak_grid_kw"]
    assert abs(hourly["demand_dynamic"].sum() - hourly["demand_flat"].sum()) < 1e-6
