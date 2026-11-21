
"""Run: streamlit run app/demo_streamlit.py (from the project root)"""

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


# ============================================================
# SMART GRID BRANDING
# ============================================================

NAVY = "#0B2D50"
BLUE = "#087CC1"
CYAN = "#00AEEF"
GREEN = "#28B83D"
GREEN_DARK = "#159B35"
BG_LIGHT = "#F2F8FC"
BORDER = "#DCEAF2"
WHITE = "#FFFFFF"

LOGO = Path(__file__).resolve().parent / "assets" / "logo_sonelgaz.png"
logo_img = Image.open(LOGO) if LOGO.exists() else None


st.set_page_config(
    page_title="Smart Grid - Dynamic Energy Management",
    page_icon=logo_img or "⚡",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(f"""
<style>

:root {{
    color-scheme: light;
}}

/* Main background */

.stApp {{
    background-color: {BG_LIGHT};
}}

/* Titles */

h1, h2, h3 {{
    color: {NAVY};
}}

/* Main header */

.sg-header {{
    border-bottom: 4px solid {GREEN};
    margin-bottom: 1rem;
    padding-bottom: .6rem;
}}

.sg-header h1 {{
    margin: 0;
    padding: 0;
    color: {NAVY};
    font-weight: 800;
}}

.sg-sub {{
    color: {BLUE};
    opacity: .85;
    margin: 0;
    font-size: 15px;
}}

/* Metric cards */

div[data-testid="stMetric"] {{
    background: {WHITE};
    border: 1px solid {BORDER};
    border-left: 5px solid {GREEN};
    border-radius: 12px;
    padding: .8rem 1rem;
    box-shadow: 0 2px 8px rgba(11,45,80,.05);
}}

div[data-testid="stMetricLabel"] p {{
    color: {NAVY};
    font-weight: 600;
}}

div[data-testid="stMetricValue"] {{
    color: {BLUE};
    font-weight: 700;
}}

/* Sidebar */

section[data-testid="stSidebar"] {{
    background-color: {WHITE};
    border-right: 2px solid {BORDER};
}}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: {NAVY};
    font-weight: 700;
}}

/* Sidebar labels */

section[data-testid="stSidebar"] label {{
    color: {NAVY};
}}

/* Buttons */

.stButton button {{
    background-color: {GREEN};
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}}

.stButton button:hover {{
    background-color: {GREEN_DARK};
    color: white;
    border: none;
}}

/* Expanders */

div[data-testid="stExpander"] {{
    background-color: {WHITE};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}

/* Dataframes */

div[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 10px;
}}

/* Links */

a {{
    color: {BLUE};
}}

/* Slider accent */

.stSlider [data-baseweb="slider"] div {{
    background-color: {GREEN};
}}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOGO AND PAGE HEADER
# ============================================================

if logo_img is not None:
    st.logo(logo_img, size="large")

    c_logo, c_title = st.columns(
        [1, 7],
        vertical_alignment="center"
    )

    c_logo.image(logo_img, width=95)

else:
    c_title = st.container()


c_title.markdown(
    '<div class="sg-header">'
    '<h1>Smart Grid Micro-Grid: Dynamic Pricing Demo</h1>'
    '<p class="sg-sub">'
    'Consumption Forecast (A) + Solar Forecast (B) + Hourly Price Agent (C)'
    '</p>'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_all():

    solar = load_solar()

    fc, act = forecast_tables(
        load_pv_predictions(),
        "test"
    )

    return (
        build_dataset(
            load_daily(),
            load_weather(),
            load_holidays(),
            solar
        ),
        pv_by_day(solar),
        fc,
        act
    )


# ============================================================
# PRICING AGENT
# ============================================================

@st.cache_resource
def get_agent(_data, _pv, _profile):

    path = MODELS / "pricing_agent.joblib"

    if path.exists():
        return PricingAgent.load(path)

    X, y = build_training_set(
        time_split(_data)[0],
        _pv,
        _profile
    )

    return PricingAgent().fit(X, y)


# ============================================================
# INITIALIZE PROJECT
# ============================================================

data, pv_table, fc_table, act_table = load_all()

profile, profile_src = get_profile()

agent = get_agent(
    data,
    pv_table,
    profile
)

_, test_df = time_split(data)


# ============================================================
# SIDEBAR - SCENARIO SETTINGS
# ============================================================

with st.sidebar:

    st.header("Scenario")

    use_fc = st.radio(
        "PV source",
        [
            "Recorded (Part B data)",
            "B's LSTM forecast (test days)"
        ]
    ).startswith("B's")

    days_ok = (
        test_df["day"][test_df["day"].isin(fc_table.index)]
        if use_fc
        else data["day"]
    )

    day = st.selectbox(
        "Day",
        days_ok.dt.date.tolist(),
        index=len(days_ok) - 1
    )

    if not use_fc and pd.Timestamp(day) not in set(test_df["day"]):
        st.caption(
            "Day is in Part A's training period (in-sample)."
        )

    st.subheader("Community")

    cfg = CommunityConfig(

        n_households=st.slider(
            "Households",
            20, 500, 100, 10
        ),

        pv_units_per_household=st.slider(
            "PV units per household",
            0.0, 10.0, 3.0, 0.5
        ),

        flex_share=st.slider(
            "Flexible share of demand",
            0.0, 0.5, 0.2, 0.05
        ),

        sensitivity=st.slider(
            "Price sensitivity",
            0.0, 1.0, 0.5, 0.1
        )
    )

    st.subheader("What-if (Part A model)")

    dT = st.slider(
        "Temperature change (deg C)",
        -10, 10, 0
    )

    force_holiday = st.checkbox(
        "Force bank holiday"
    )


# ============================================================
# CONSUMPTION PREDICTION - PART A
# ============================================================

row = data[data["day"].dt.date == day].iloc[0].copy()

row["temperatureMax"] += dT
row["temperatureMin"] += dT

if force_holiday:
    row["is_holiday"] = 1

pred = float(
    predict_baseline_consumption(
        pd.DataFrame([row[FEATURES]])
    )[0]
)


# ============================================================
# SOLAR PV - PART B
# ============================================================

ts = pd.Timestamp(day)

actual = (
    act_table.loc[ts].to_numpy()
    if use_fc
    else pv_table.loc[ts].to_numpy()
)

fc_h = (
    fc_table.loc[ts].to_numpy()
    if use_fc
    else None
)


# ============================================================
# SIMULATION - PART C
# ============================================================

hourly, s = simulate_day(
    pred,
    bool(row["is_weekend"]),
    actual,
    agent,
    profile,
    cfg,
    fc_h
)

f, d = s["flat"], s["dynamic"]


# ============================================================
# KEY PERFORMANCE INDICATORS
# ============================================================

st.subheader("Smart Grid Performance")

c = st.columns(4)

c[0].metric(
    "Predicted kWh / household",
    f"{pred:.2f}",
    f"actual {row['kwh_per_household']:.2f}",
    delta_color="off"
)

c[1].metric(
    "Peak grid import (kW)",
    f"{d['peak_grid_kw']:.0f}",
    f"{d['peak_grid_kw'] - f['peak_grid_kw']:+.0f} vs flat",
    delta_color="inverse"
)

c[2].metric(
    "PV self-use (%)",
    f"{d['pv_self_use_pct']:.0f}",
    f"{d['pv_self_use_pct'] - f['pv_self_use_pct']:+.1f} pts"
)

c[3].metric(
    "Community cost (GBP)",
    f"{d['cost_gbp']:.0f}",
    f"{d['cost_gbp'] - f['cost_gbp']:+.0f} vs flat",
    delta_color="inverse"
)


# ============================================================
# CHARTS
# ============================================================

l, r = st.columns(2)

l.subheader("Power balance (kWh per hour)")

_cols = (
    ["demand_flat", "demand_dynamic", "pv"]
    + (["pv_forecast"] if use_fc else [])
)

_palette = {
    "demand_flat": "#AAB7C4",
    "demand_dynamic": NAVY,
    "pv": GREEN,
    "pv_forecast": CYAN
}

l.line_chart(
    hourly.set_index("hour")[_cols],
    color=[_palette[col] for col in _cols]
)


r.subheader("Price set by the agent (p/kWh)")

r.bar_chart(
    hourly.set_index("hour")["price"],
    color=GREEN
)


# ============================================================
# HOURLY DETAILS
# ============================================================

with st.expander("Hourly details"):

    st.dataframe(
        hourly.round(2),
        width="stretch"
    )


# ============================================================
# INFORMATION
# ============================================================

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
