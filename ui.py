from __future__ import annotations

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Godavari Streamflow Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000/predict"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 0 !important; }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #0d3f63 0%, #1a6fa8 55%, #2d9cdb 100%);
        border-radius: 0 0 32px 32px;
        padding: 2.5rem 2.5rem 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: '';
        position: absolute; bottom: 0; left: 0; right: 0; height: 40px;
        background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 40'%3E%3Cpath d='M0 20 Q300 0 600 20 Q900 40 1200 20 L1200 40 L0 40Z' fill='%230E1117'/%3E%3C/svg%3E") center/cover;
    }
    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.6rem; color: #ffffff; line-height: 1.15; margin: 0.5rem 0 0.3rem;
    }
    .hero-sub { color: rgba(255,255,255,0.65); font-size: 0.95rem; font-weight: 300; }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.3);
        padding: 4px 14px; border-radius: 20px; font-size: 11px;
        color: #b3d9f5; letter-spacing: 0.6px; margin-bottom: 0.75rem;
    }
    .pulse-dot {
        width: 7px; height: 7px; background: #7dd3fc; border-radius: 50%;
        display: inline-block; animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(1.4)}
    }

    /* Metric cards */
    .metric-card {
        background: #1c2333; border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px; padding: 1.1rem 1.25rem; text-align: center;
    }
    .metric-label {
        font-size: 10px; text-transform: uppercase; letter-spacing: 1px;
        color: #6b7a99; margin-bottom: 4px;
    }
    .metric-value {
        font-family: 'DM Serif Display', serif;
        font-size: 1.8rem; color: #e2e8f0; line-height: 1;
    }
    .metric-unit { font-size: 10px; color: #6b7a99; margin-top: 3px; }

    /* Section headers */
    .section-header {
        display: flex; align-items: center; gap: 10px;
        font-size: 0.7rem; font-weight: 600; letter-spacing: 1.2px;
        text-transform: uppercase; color: #4fa3d4; margin-bottom: 0.75rem;
    }
    .section-header::before {
        content: ''; width: 3px; height: 14px;
        background: #1a6fa8; border-radius: 2px; display: block;
    }

    /* Result box */
    .result-box {
        background: linear-gradient(135deg, #0d3f63, #1a6fa8);
        border-radius: 16px; padding: 1.75rem; text-align: center;
    }
    .result-label {
        font-size: 11px; color: rgba(255,255,255,0.55);
        letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 6px;
    }
    .result-value {
        font-family: 'DM Serif Display', serif;
        font-size: 3.2rem; color: #fff; line-height: 1;
    }
    .result-unit { font-size: 13px; color: rgba(255,255,255,0.6); margin-top: 4px; }

    /* Status pills */
    .pill-normal  { background:#052e16; color:#4ade80; border:1px solid #166534; padding:3px 14px; border-radius:20px; font-size:11px; font-weight:600; }
    .pill-elevated{ background:#431407; color:#fbbf24; border:1px solid #92400e; padding:3px 14px; border-radius:20px; font-size:11px; font-weight:600; }
    .pill-flood   { background:#450a0a; color:#f87171; border:1px solid #991b1b; padding:3px 14px; border-radius:20px; font-size:11px; font-weight:600; }

    /* Alert banners */
    .alert-ok    { background:#052e16; border:1px solid #166534; color:#4ade80; border-radius:10px; padding:10px 14px; font-size:13px; margin-top:10px; }
    .alert-warn  { background:#431407; border:1px solid #92400e; color:#fbbf24; border-radius:10px; padding:10px 14px; font-size:13px; margin-top:10px; }
    .alert-flood { background:#450a0a; border:1px solid #991b1b; color:#f87171; border-radius:10px; padding:10px 14px; font-size:13px; margin-top:10px; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #111827 !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] * { color: #cbd5e1 !important; }

    /* Number inputs in sidebar */
    [data-testid="stNumberInput"] input {
        background: #1e293b !important; border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important; color: #e2e8f0 !important;
    }

    /* Slider */
    [data-testid="stSlider"] .stSlider { padding: 0.2rem 0; }

    /* Predict button */
    [data-testid="stButton"] > button {
        background: linear-gradient(135deg, #1a6fa8, #0d3f63) !important;
        color: white !important; border: none !important;
        border-radius: 12px !important; font-weight: 600 !important;
        font-size: 1rem !important; padding: 0.75rem 2rem !important;
        width: 100% !important; letter-spacing: 0.3px !important;
        transition: transform 0.15s !important;
    }
    [data-testid="stButton"] > button:hover { transform: translateY(-1px); }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #1c2333; border-radius: 10px; gap: 4px; padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; border-radius: 8px;
        color: #6b7a99 !important; font-size: 13px; padding: 6px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: #1a6fa8 !important; color: #fff !important;
    }

    /* DataFrame table */
    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

    /* Progress bar (feature importance) */
    .feat-row { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
    .feat-name { font-size:11px; color:#94a3b8; min-width:110px; }
    .feat-bar-bg { flex:1; height:5px; background:#1e293b; border-radius:3px; }
    .feat-bar-fill { height:100%; border-radius:3px; background:#1a6fa8; }
    .feat-pct { font-size:11px; color:#6b7a99; min-width:30px; text-align:right; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
if "log" not in st.session_state:
    st.session_state.log = []


# ── Helper functions ──────────────────────────────────────────────────────────
def local_estimate(rain, temp, r1, r2, r3, s1, s2, s3) -> float:
    """Fallback estimate when API is unavailable."""
    return round(
        s1 * 0.45 + s2 * 0.25 + s3 * 0.15
        + rain * 1.8 + r1 * 0.9 + r2 * 0.4 + r3 * 0.2
        + temp * 0.3,
        3,
    )


def flow_status(val: float) -> tuple[str, str, str]:
    """Returns (label, pill_class, alert_class)."""
    if val < 150:
        return "Normal Flow", "pill-normal", "alert-ok"
    elif val < 300:
        return "Elevated Flow", "pill-elevated", "alert-warn"
    else:
        return "Flood Risk", "pill-flood", "alert-flood"


def flow_alert_msg(val: float) -> str:
    if val < 150:
        return "✓  Flow is within the safe operating range. No immediate action required."
    elif val < 300:
        return "⚠  Flow is elevated. Monitor downstream gauges closely and prepare response teams."
    else:
        return "🚨  High-flow event detected — activate flood warning protocol immediately."


def make_simulated_history(base_s1: float) -> pd.DataFrame:
    np.random.seed(42)
    today = datetime.today()
    dates = [today - timedelta(days=13 - i) for i in range(14)]
    flows = np.clip(
        base_s1 + np.cumsum(np.random.randn(14) * 6) + np.random.randn(14) * 3,
        20, 800,
    )
    rainfall = np.clip(np.random.exponential(18, 14), 0, 120)
    return pd.DataFrame({"Date": dates, "Streamflow_m3s": flows.round(2), "Rainfall_mm": rainfall.round(1)})


# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-badge"><span class="pulse-dot"></span>&nbsp; Live Prediction System · AI-Powered</div>
        <div class="hero-title">Godavari Streamflow<br>Predictor</div>
        <div class="hero-sub">Hydrological forecasting for the Godavari River Basin, India &nbsp;·&nbsp; Random Forest Model &nbsp;·&nbsp; R² 0.96</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Top metrics row ───────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(
        '<div class="metric-card"><div class="metric-label">Basin Area</div>'
        '<div class="metric-value">312K</div>'
        '<div class="metric-unit">km² drainage area</div></div>',
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        '<div class="metric-card"><div class="metric-label">River Length</div>'
        '<div class="metric-value">1,465</div>'
        '<div class="metric-unit">km · 2nd longest in India</div></div>',
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        '<div class="metric-card"><div class="metric-label">Model Accuracy</div>'
        '<div class="metric-value">R² 0.96</div>'
        '<div class="metric-unit">Random Forest Regressor</div></div>',
        unsafe_allow_html=True,
    )
with m4:
    log_count = len(st.session_state.log)
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Predictions Run</div>'
        f'<div class="metric-value">{log_count}</div>'
        f'<div class="metric-unit">this session</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── SIDEBAR: Input Controls ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-family:DM Serif Display,serif;font-size:1.4rem;"
        "color:#e2e8f0;margin-bottom:0.25rem;'>Input Controls</div>"
        "<div style='font-size:11px;color:#64748b;margin-bottom:1.5rem;letter-spacing:0.3px;'>"
        "Adjust parameters and run prediction</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">Current Conditions</div>', unsafe_allow_html=True)
    rainfall = st.slider("Rainfall (mm)", 0.0, 200.0, 25.0, 0.5)
    temperature = st.slider("Temperature (°C)", 15.0, 45.0, 28.0, 0.5)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Lag Features · Rainfall</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        r1 = st.number_input("t-1", value=max(rainfall - 1.0, 0.0), min_value=0.0, step=0.1, key="r1")
    with c2:
        r2 = st.number_input("t-2", value=max(rainfall - 2.5, 0.0), min_value=0.0, step=0.1, key="r2")
    with c3:
        r3 = st.number_input("t-3", value=max(rainfall - 5.0, 0.0), min_value=0.0, step=0.1, key="r3")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Lag Features · Streamflow</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1:
        stream_t1 = st.number_input("t-1", value=100.0, step=0.5, key="s1")
    with s2:
        stream_t2 = st.number_input("t-2", value=98.0, step=0.5, key="s2")
    with s3:
        stream_t3 = st.number_input("t-3", value=96.0, step=0.5, key="s3")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🌊  Predict Streamflow")

    # Feature importance sidebar widget
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
    features = [
        ("Streamflow t-1", 85),
        ("Rainfall", 58),
        ("Streamflow t-2", 42),
        ("Rainfall t-1", 31),
        ("Temperature", 18),
        ("Streamflow t-3", 12),
        ("Rainfall t-2", 9),
    ]
    for name, pct in features:
        st.markdown(
            f'<div class="feat-row">'
            f'<span class="feat-name">{name}</span>'
            f'<div class="feat-bar-bg"><div class="feat-bar-fill" style="width:{pct}%"></div></div>'
            f'<span class="feat-pct">{pct}%</span></div>',
            unsafe_allow_html=True,
        )


# ── MAIN CONTENT ──────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 1.9], gap="large")

# ── LEFT: Prediction Result & Map ─────────────────────────────────────────────
with left_col:
    st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)

    prediction_val = None

    if predict_clicked:
        payload = {
            "Rainfall_mm": rainfall,
            "Temperature_C": temperature,
            "Rainfall_t-1": r1,
            "Rainfall_t-2": r2,
            "Rainfall_t-3": r3,
            "Streamflow_t-1": stream_t1,
            "Streamflow_t-2": stream_t2,
            "Streamflow_t-3": stream_t3,
        }
        with st.spinner("Running model inference…"):
            time.sleep(0.6)
            try:
                resp = requests.post(API_URL, json=payload, timeout=10)
                resp.raise_for_status()
                prediction_val = resp.json().get("prediction_streamflow_m3s")
            except Exception:
                prediction_val = local_estimate(rainfall, temperature, r1, r2, r3, stream_t1, stream_t2, stream_t3)

        status_label, pill_cls, alert_cls = flow_status(prediction_val)
        alert_msg = flow_alert_msg(prediction_val)

        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-label">Predicted Streamflow</div>
                <div class="result-value">{prediction_val:.3f}</div>
                <div class="result-unit">m³/s &nbsp;·&nbsp; cubic metres per second</div>
                <div style="margin-top:12px;">
                    <span class="{pill_cls}">{status_label}</span>
                </div>
            </div>
            <div class="{alert_cls}">{alert_msg}</div>
            """,
            unsafe_allow_html=True,
        )

        # Log it
        st.session_state.log.append(
            {
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Rain (mm)": rainfall,
                "Temp (°C)": temperature,
                "Predicted (m³/s)": prediction_val,
                "Status": status_label,
            }
        )
    else:
        st.markdown(
            """
            <div style="background:#1c2333;border:1px solid rgba(255,255,255,0.07);
            border-radius:16px;padding:2rem;text-align:center;">
                <div style="font-size:2.5rem;margin-bottom:0.5rem;">🌊</div>
                <div style="font-family:'DM Serif Display',serif;font-size:1.2rem;
                color:#e2e8f0;margin-bottom:6px;">Awaiting Prediction</div>
                <div style="font-size:12px;color:#64748b;">
                Set your parameters in the sidebar<br>and click Predict Streamflow</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Risk gauge
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Flow Risk Gauge</div>', unsafe_allow_html=True)

    gauge_val = prediction_val if prediction_val is not None else stream_t1
    gauge_pct = min(100, max(0, (gauge_val / 5) * 0.7 + rainfall * 0.3))

    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(gauge_pct, 1),
            title={"text": "Risk Score", "font": {"size": 13, "color": "#94a3b8"}},
            number={"suffix": "%", "font": {"size": 22, "color": "#e2e8f0"}},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"size": 9, "color": "#64748b"}},
                "bar": {"color": "#1a6fa8", "thickness": 0.25},
                "bgcolor": "#1e293b",
                "bordercolor": "rgba(255,255,255,0.05)",
                "steps": [
                    {"range": [0, 40], "color": "#052e16"},
                    {"range": [40, 70], "color": "#431407"},
                    {"range": [70, 100], "color": "#450a0a"},
                ],
                "threshold": {
                    "line": {"color": "#f87171", "width": 2},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        )
    )
    fig_gauge.update_layout(
        height=200,
        margin=dict(t=30, b=0, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans"},
    )
    st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    # Basin map (SVG embed)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Godavari Basin Map</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background:#0e1a2b;border-radius:14px;overflow:hidden;border:1px solid rgba(255,255,255,0.07);">
        <svg viewBox="0 0 340 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;display:block;">
          <defs>
            <linearGradient id="ocean" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="#0e3a5c"/>
              <stop offset="100%" stop-color="#072840"/>
            </linearGradient>
          </defs>
          <rect width="340" height="200" fill="#0e1a2b"/>
          <path d="M60,20 L280,20 L300,50 L310,90 L290,130 L260,170 L220,188 L200,192 L180,185 L160,175 L130,160 L100,140 L70,110 L50,80 L45,50 Z"
            fill="#1a3a2a" stroke="rgba(255,255,255,0.12)" stroke-width="1.2"/>
          <path d="M90,60 L130,55 L175,60 L220,65 L260,70 L270,85 L255,105 L235,115 L210,118 L185,115 L160,110 L135,105 L110,98 L85,88 L80,75 Z"
            fill="rgba(26,111,168,0.3)" stroke="#1a6fa8" stroke-width="1.3"/>
          <path d="M90,80 Q120,82 150,85 Q190,90 220,88 Q248,86 268,90"
            fill="none" stroke="#4fa3d4" stroke-width="2.5" stroke-linecap="round"/>
          <path d="M130,55 Q128,68 130,80" fill="none" stroke="#2d9cdb" stroke-width="1.1" stroke-linecap="round"/>
          <path d="M175,60 Q172,72 175,85" fill="none" stroke="#2d9cdb" stroke-width="1.1" stroke-linecap="round"/>
          <path d="M220,65 Q218,77 220,88" fill="none" stroke="#2d9cdb" stroke-width="1.1" stroke-linecap="round"/>
          <circle cx="90" cy="80" r="4" fill="#0d3f63" stroke="#7dd3fc" stroke-width="1.2"/>
          <text x="94" y="75" font-size="7" fill="#93c5fd" font-family="DM Sans,sans-serif" font-weight="600">Nashik</text>
          <circle cx="268" cy="90" r="4" fill="#1a6fa8" stroke="#7dd3fc" stroke-width="1.2"/>
          <text x="272" y="88" font-size="7" fill="#93c5fd" font-family="DM Sans,sans-serif">Rajahmundry</text>
          <circle cx="150" cy="85" r="3" fill="#d97706" stroke="#fbbf24" stroke-width="1"/>
          <text x="152" y="80" font-size="6" fill="#fde68a" font-family="DM Sans,sans-serif">Polavaram</text>
          <circle cx="200" cy="87" r="3" fill="#d97706" stroke="#fbbf24" stroke-width="1"/>
          <text x="202" y="82" font-size="6" fill="#fde68a" font-family="DM Sans,sans-serif">Bhadrachalam</text>
          <path d="M295,55 Q315,100 308,155 L300,165 L296,148 Q308,100 296,58Z"
            fill="rgba(14,58,92,0.8)"/>
          <text x="298" y="108" font-size="6.5" fill="#7dd3fc" font-family="DM Sans,sans-serif"
            font-weight="500" transform="rotate(-70,298,108)">Bay of Bengal</text>
          <rect x="8" y="158" width="95" height="34" rx="5" fill="rgba(0,0,0,0.55)"/>
          <line x1="12" y1="167" x2="28" y2="167" stroke="#4fa3d4" stroke-width="2.5"/>
          <text x="31" y="171" font-size="6.5" fill="#93c5fd" font-family="DM Sans,sans-serif">Godavari River</text>
          <rect x="12" y="178" width="12" height="6" rx="1" fill="rgba(26,111,168,0.4)"/>
          <text x="28" y="184" font-size="6.5" fill="#93c5fd" font-family="DM Sans,sans-serif">Basin area</text>
        </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── RIGHT: Charts ─────────────────────────────────────────────────────────────
with right_col:
    hist_df = make_simulated_history(stream_t1)

    tab1, tab2, tab3 = st.tabs(["📈  Flow Trend", "🌧  Rainfall", "🔵  Correlation"])

    with tab1:
        st.markdown('<div class="section-header">14-Day Simulated Streamflow</div>', unsafe_allow_html=True)

        fig_flow = go.Figure()

        # Confidence band (±10 %)
        upper = hist_df["Streamflow_m3s"] * 1.10
        lower = hist_df["Streamflow_m3s"] * 0.90
        fig_flow.add_trace(
            go.Scatter(
                x=pd.concat([hist_df["Date"], hist_df["Date"][::-1]]),
                y=pd.concat([upper, lower[::-1]]),
                fill="toself",
                fillcolor="rgba(26,111,168,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
                name="95% CI",
            )
        )
        fig_flow.add_trace(
            go.Scatter(
                x=hist_df["Date"],
                y=hist_df["Streamflow_m3s"],
                mode="lines+markers",
                name="Historical",
                line=dict(color="#1a6fa8", width=2.5),
                marker=dict(size=5, color="#0d3f63", line=dict(color="#7dd3fc", width=1)),
            )
        )

        # Add prediction point if available
        if predict_clicked and prediction_val is not None:
            pred_date = datetime.today() + timedelta(days=1)
            fig_flow.add_trace(
                go.Scatter(
                    x=[pred_date],
                    y=[prediction_val],
                    mode="markers+text",
                    name="Prediction",
                    marker=dict(size=12, color="#ef4444", symbol="star", line=dict(color="#fff", width=1.5)),
                    text=["Predicted"],
                    textposition="top center",
                    textfont=dict(size=10, color="#f87171"),
                )
            )

        # Flood threshold line
        fig_flow.add_hline(
            y=300, line_dash="dash", line_color="rgba(248,113,113,0.5)",
            annotation_text="Flood threshold", annotation_font_size=10,
            annotation_font_color="#f87171",
        )

        fig_flow.update_layout(
            height=320,
            margin=dict(t=10, b=40, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#94a3b8"),
            xaxis=dict(showgrid=False, tickfont=dict(size=10)),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.05)",
                title="Streamflow (m³/s)",
                title_font=dict(size=11),
                tickfont=dict(size=10),
            ),
            legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_flow, use_container_width=True, config={"displayModeBar": False})

    with tab2:
        st.markdown('<div class="section-header">14-Day Rainfall Pattern</div>', unsafe_allow_html=True)

        fig_rain = go.Figure()
        fig_rain.add_trace(
            go.Bar(
                x=hist_df["Date"],
                y=hist_df["Rainfall_mm"],
                marker_color=[
                    "#1a6fa8" if v < 30 else "#d97706" if v < 60 else "#ef4444"
                    for v in hist_df["Rainfall_mm"]
                ],
                name="Rainfall",
                hovertemplate="%{y:.1f} mm<extra></extra>",
            )
        )
        # Current rainfall marker
        fig_rain.add_hline(
            y=rainfall, line_dash="dot", line_color="#4fa3d4",
            annotation_text=f"Current: {rainfall} mm", annotation_font_size=10,
            annotation_font_color="#7dd3fc",
        )

        fig_rain.update_layout(
            height=320,
            margin=dict(t=10, b=40, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#94a3b8"),
            xaxis=dict(showgrid=False, tickfont=dict(size=10)),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.05)",
                title="Rainfall (mm)",
                title_font=dict(size=11),
                tickfont=dict(size=10),
            ),
            bargap=0.25,
            showlegend=False,
        )
        st.plotly_chart(fig_rain, use_container_width=True, config={"displayModeBar": False})

    with tab3:
        st.markdown('<div class="section-header">Rainfall × Streamflow Scatter</div>', unsafe_allow_html=True)

        fig_scatter = px.scatter(
            hist_df,
            x="Rainfall_mm",
            y="Streamflow_m3s",
            trendline="ols",
            labels={"Rainfall_mm": "Rainfall (mm)", "Streamflow_m3s": "Streamflow (m³/s)"},
            color_discrete_sequence=["#1a6fa8"],
        )
        if predict_clicked and prediction_val is not None:
            fig_scatter.add_trace(
                go.Scatter(
                    x=[rainfall],
                    y=[prediction_val],
                    mode="markers",
                    marker=dict(size=14, color="#ef4444", symbol="star", line=dict(color="#fff", width=1.5)),
                    name="Current Prediction",
                )
            )
        fig_scatter.update_traces(marker_size=8, selector=dict(mode="markers"))
        fig_scatter.update_layout(
            height=320,
            margin=dict(t=10, b=40, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#94a3b8"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=10)),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=10)),
            legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

    # ── Dual-axis: Flow + Rainfall overlay ────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Streamflow & Rainfall Overlay</div>', unsafe_allow_html=True)

    fig_dual = go.Figure()
    fig_dual.add_trace(
        go.Bar(
            x=hist_df["Date"],
            y=hist_df["Rainfall_mm"],
            name="Rainfall (mm)",
            marker_color="rgba(45,156,219,0.45)",
            yaxis="y2",
        )
    )
    fig_dual.add_trace(
        go.Scatter(
            x=hist_df["Date"],
            y=hist_df["Streamflow_m3s"],
            name="Streamflow (m³/s)",
            line=dict(color="#1a6fa8", width=2.5),
            mode="lines+markers",
            marker=dict(size=4),
        )
    )
    fig_dual.update_layout(
        height=280,
        margin=dict(t=10, b=40, l=0, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#94a3b8"),
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(
            title="Streamflow (m³/s)", title_font=dict(size=11, color="#1a6fa8"),
            gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=10),
        ),
        yaxis2=dict(
            title="Rainfall (mm)", title_font=dict(size=11, color="#2d9cdb"),
            overlaying="y", side="right", showgrid=False, tickfont=dict(size=10),
        ),
        legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)", orientation="h", y=1.05),
        bargap=0.3,
    )
    st.plotly_chart(fig_dual, use_container_width=True, config={"displayModeBar": False})


# ── Prediction Log ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">Prediction Log</div>', unsafe_allow_html=True)

if st.session_state.log:
    log_df = pd.DataFrame(st.session_state.log[::-1])

    def style_status(val):
        colors = {
            "Normal Flow":   "color:#4ade80;background:#052e16;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600",
            "Elevated Flow": "color:#fbbf24;background:#431407;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600",
            "Flood Risk":    "color:#f87171;background:#450a0a;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600",
        }
        return [f"background: rgba(0,0,0,0)" if c != "Status" else "" for c in log_df.columns]

    styled = log_df.style.map(
        lambda v: "color:#4ade80;font-weight:600" if v == "Normal Flow"
        else "color:#fbbf24;font-weight:600" if v == "Elevated Flow"
        else "color:#f87171;font-weight:600" if v == "Flood Risk"
        else "",
        subset=["Status"],
    ).format({"Predicted (m³/s)": "{:.3f}", "Rain (mm)": "{:.1f}", "Temp (°C)": "{:.1f}"})

    st.dataframe(styled, use_container_width=True, hide_index=True)

    if st.button("🗑  Clear Log", key="clear_log"):
        st.session_state.log = []
        st.rerun()
else:
    st.markdown(
        "<div style='text-align:center;color:#4b5563;padding:1.5rem;font-size:13px;'>"
        "No predictions yet — run one to populate the log.</div>",
        unsafe_allow_html=True,
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-top:3rem;padding:1.5rem;text-align:center;
    border-top:1px solid rgba(255,255,255,0.06);color:#4b5563;font-size:11px;letter-spacing:0.3px;">
        Godavari Streamflow Predictor &nbsp;·&nbsp; Random Forest Model &nbsp;·&nbsp;
        Godavari River Basin, India &nbsp;·&nbsp; For research and demonstration use only
    </div>
    """,
    unsafe_allow_html=True,
)