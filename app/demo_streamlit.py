"""Run:  streamlit run app/demo_streamlit.py   (from the project root)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import streamlit as st
from PIL import Image
from src.community import CommunityConfig, pv_by_day
from src.data_loader import load_daily, load_holidays, load_solar, load_weather
from src.load_profile import get_profile
from src.model_regression import predict_baseline_consumption
from src.preprocessing import FEATURES, build_dataset, time_split
from src.pricing_agent import PricingAgent, build_training_set
from src.pv_forecast import forecast_tables, load_pv_predictions
from src.simulation import simulate_day
from src.utils import MODELS

# ---- SONELGAZ branding (logo + colours taken from the logo) ----
BLUE, ORANGE, ORANGE_DEEP, NAVY = "#0168C9", "#FEA702", "#FE8901", "#0B2545"
LOGO = Path(__file__).resolve().parent / "assets" / "logo_sonelgaz.png"
logo_img = Image.open(LOGO) if LOGO.exists() else None

st.set_page_config(page_title="SONELGAZ - Smart Grid dynamic pricing", page_icon=logo_img or "⚡", layout="wide")

st.markdown(f"""
<style>
:root {{ color-scheme: light; }}
h1, h2, h3 {{ color: {BLUE}; }}
.sg-header {{ border-bottom: 4px solid {ORANGE}; margin-bottom: 1rem; padding-bottom: .4rem; }}
.sg-header h1 {{ margin: 0; padding: 0; color: {BLUE}; }}
.sg-sub {{ color: {NAVY}; opacity: .75; margin: 0; }}
div[data-testid="stMetric"] {{ background: #FFF4E0; border-left: 6px solid {ORANGE}; border-radius: 8px; padding: .6rem .9rem; }}
div[data-testid="stMetricLabel"] p {{ color: {BLUE}; font-weight: 600; }}
section[data-testid="stSidebar"] {{ border-right: 3px solid {ORANGE}; }}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{ color: {BLUE}; }}
</style>
""", unsafe_allow_html=True)

if logo_img is not None:
    st.logo(logo_img, size="large")          # logo at the top of the sidebar / header
    c_logo, c_title = st.columns([1, 7], vertical_alignment="center")
    c_logo.image(logo_img, width=95)
else:
    c_title = st.container()
c_title.markdown('<div class="sg-header"><h1>Smart Grid micro-grid: dynamic pricing demo</h1>'
                 '<p class="sg-sub">Consumption forecast (A) + solar forecast (B) + hourly price agent (C)</p></div>',
                 unsafe_allow_html=True)

@st.cache_data
def load_all():
    solar = load_solar()
    fc, act = forecast_tables(load_pv_predictions(), 'test')
    return build_dataset(load_daily(), load_weather(), load_holidays(), solar), pv_by_day(solar), fc, act

@st.cache_resource
def get_agent(_data, _pv, _profile):
    path = MODELS / "pricing_agent.joblib"
    if path.exists(): return PricingAgent.load(path)
    X, y = build_training_set(time_split(_data)[0], _pv, _profile)
    return PricingAgent().fit(X, y)

data, pv_table, fc_table, act_table = load_all()
profile, profile_src = get_profile()
agent = get_agent(data, pv_table, profile)
_, test_df = time_split(data)

with st.sidebar:
    st.header("Scenario")
    use_fc = st.radio("PV source", ["Recorded (Part B data)", "B's LSTM forecast (test days)"]) .startswith("B's")
    days_ok = test_df["day"][test_df["day"].isin(fc_table.index)] if use_fc else data["day"]
    day = st.selectbox("Day", days_ok.dt.date.tolist(), index=len(days_ok) - 1)
    if not use_fc and pd.Timestamp(day) not in set(test_df["day"]):
        st.caption("Day is in Part A's training period (in-sample).")
    st.subheader("Community")
    cfg = CommunityConfig(
        n_households=st.slider("Households", 20, 500, 100, 10),
        pv_units_per_household=st.slider("PV units per household", 0.0, 10.0, 3.0, 0.5),
        flex_share=st.slider("Flexible share of demand", 0.0, 0.5, 0.2, 0.05),
        sensitivity=st.slider("Price sensitivity", 0.0, 1.0, 0.5, 0.1))
    st.subheader("What-if (Part A model)")
    dT = st.slider("Temperature change (deg C)", -10, 10, 0)
    force_holiday = st.checkbox("Force bank holiday")

row = data[data["day"].dt.date == day].iloc[0].copy()
row["temperatureMax"] += dT; row["temperatureMin"] += dT
if force_holiday: row["is_holiday"] = 1
pred = float(predict_baseline_consumption(pd.DataFrame([row[FEATURES]]))[0])

ts = pd.Timestamp(day)
actual = act_table.loc[ts].to_numpy() if use_fc else pv_table.loc[ts].to_numpy()
fc_h = fc_table.loc[ts].to_numpy() if use_fc else None
hourly, s = simulate_day(pred, bool(row["is_weekend"]), actual, agent, profile, cfg, fc_h)
f, d = s["flat"], s["dynamic"]

c = st.columns(4)
c[0].metric("Predicted kWh / household", f"{pred:.2f}", f"actual {row['kwh_per_household']:.2f}", delta_color="off")
c[1].metric("Peak grid import (kW)", f"{d['peak_grid_kw']:.0f}", f"{d['peak_grid_kw']-f['peak_grid_kw']:+.0f} vs flat", delta_color="inverse")
c[2].metric("PV self-use (%)", f"{d['pv_self_use_pct']:.0f}", f"{d['pv_self_use_pct']-f['pv_self_use_pct']:+.1f} pts")
c[3].metric("Community cost (GBP)", f"{d['cost_gbp']:.0f}", f"{d['cost_gbp']-f['cost_gbp']:+.0f} vs flat", delta_color="inverse")

l, r = st.columns(2)
l.subheader("Power balance (kWh per hour)")
_cols = ["demand_flat", "demand_dynamic", "pv"] + (["pv_forecast"] if use_fc else [])
_palette = {"demand_flat": "#8A94A6", "demand_dynamic": BLUE, "pv": ORANGE, "pv_forecast": ORANGE_DEEP}
l.line_chart(hourly.set_index("hour")[_cols], color=[_palette[c] for c in _cols])
r.subheader("Price set by the agent (p/kWh)")
r.bar_chart(hourly.set_index("hour")["price"], color=ORANGE)
with st.expander("Hourly details"):
    st.dataframe(hourly.round(2), width="stretch")
if use_fc:
    st.caption(
        f"Hourly shape: {profile_src}. "
        "Part B LSTM forecasts are used for pricing decisions, "
        "while actual PV values are used for evaluation."
    )
else:
    st.caption(
        f"Hourly shape: {profile_src}. "
        "Recorded PV data is used for pricing decisions and evaluation."
    )
