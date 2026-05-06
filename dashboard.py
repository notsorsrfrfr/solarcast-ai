import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
import random
from datetime import datetime, timedelta

st.set_page_config(page_title="SolarCast AI", page_icon=None, layout="wide")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0a0f1e; }
    [data-testid="stHeader"] { background-color: #0a0f1e; }
    .block-container { padding-top: 2rem; }
    .sc-header { font-size: 28px; font-weight: 600; color: #e8f4f0; letter-spacing: -0.5px; margin-bottom: 2px; }
    .sc-sub { font-size: 13px; color: #4a7c6f; margin-bottom: 0; }
    .sc-badge { display: inline-block; background: #0d2e26; color: #00c896; border: 1px solid #00c896;
        border-radius: 4px; font-size: 11px; font-weight: 600; padding: 2px 8px;
        letter-spacing: 0.08em; text-transform: uppercase; margin-left: 10px; vertical-align: middle; }
    .kpi-card { background: #0d1425; border: 1px solid #1a2a4a; border-radius: 10px; padding: 18px 20px; }
    .kpi-label { font-size: 11px; font-weight: 600; color: #3a5a8a; text-transform: uppercase;
        letter-spacing: 0.1em; margin-bottom: 6px; }
    .kpi-value { font-size: 26px; font-weight: 600; color: #e8f4f0; line-height: 1; }
    .kpi-unit { font-size: 13px; color: #4a7c6f; margin-left: 4px; }
    .kpi-delta { font-size: 12px; color: #00c896; margin-top: 4px; }
    .kpi-delta.warn { color: #f0a040; }
    .alert-box { background: #1a0d0d; border-left: 3px solid #c0392b;
        border-radius: 6px; padding: 12px 14px; margin-bottom: 8px; }
    .alert-title { font-size: 12px; font-weight: 600; color: #e05c5c; margin-bottom: 4px; }
    .alert-detail { font-size: 12px; color: #8a6a6a; line-height: 1.6; }
    .section-title { font-size: 13px; font-weight: 600; color: #3a5a8a;
        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px; }
    div[data-testid="stMetric"] { background: #0d1425; border: 1px solid #1a2a4a; border-radius: 10px; padding: 14px 18px; }
    div[data-testid="stMetricLabel"] { color: #3a5a8a !important; font-size: 11px !important; }
    div[data-testid="stMetricValue"] { color: #e8f4f0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Model runs directly here, no API needed ──────────────────────────────────

@st.cache_resource(show_spinner="Training forecast model...")
def load_model():
    from prophet import Prophet

    # Generate Karnataka solar data
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=365 * 24, freq='h')
    df = pd.DataFrame({'ds': dates})
    hour = df['ds'].dt.hour
    month = df['ds'].dt.month
    daily = np.exp(-0.5 * ((hour - 13) / 3.5) ** 2)
    seasonal = 0.75 + 0.25 * np.sin((month - 3) * np.pi / 6)
    noise = np.random.normal(1.0, 0.15, len(df)).clip(0.2, 1.3)
    df['y'] = (daily * seasonal * noise * 850).clip(0).round(2)

    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        interval_width=0.90
    )
    m.fit(df)
    return m, df


@st.cache_data(show_spinner="Generating 72-hour forecast...")
def get_forecast(_model):
    future = _model.make_future_dataframe(periods=72, freq='h')
    forecast = _model.predict(future)
    result = forecast.tail(72)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    result['yhat'] = result['yhat'].clip(0).round(2)
    result['yhat_lower'] = result['yhat_lower'].clip(0).round(2)
    result['yhat_upper'] = result['yhat_upper'].clip(0).round(2)
    return result


def get_anomalies(df, threshold=0.35):
    df = df.copy()
    df['hour'] = pd.to_datetime(df['ds']).dt.hour
    hourly_mean = df.groupby('hour')['y'].mean()
    df['expected'] = df['hour'].map(hourly_mean)
    df['drop_pct'] = (df['expected'] - df['y']) / (df['expected'] + 1)
    anomalies = df[df['drop_pct'] > threshold].tail(20)
    return [{
        "timestamp": str(row['ds']),
        "actual_mw": round(row['y'], 2),
        "expected_mw": round(row['expected'], 2),
        "drop_pct": round(row['drop_pct'] * 100, 1)
    } for _, row in anomalies.iterrows()]


def get_current():
    hour = datetime.now().hour
    base = 850.0 * math.exp(-0.5 * ((hour - 13) / 3.5) ** 2)
    current = round(max(base * random.uniform(0.85, 1.1), 0.0), 1)
    return {
        "current_mw": current,
        "capacity_mw": 850.0,
        "utilization_pct": round((current / 850.0) * 100, 1),
        "status": "normal" if current > 100 else "low",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M')
    }


# ── Load everything ───────────────────────────────────────────────────────────

model, df = load_model()
forecast_df = get_forecast(model)
anomalies = get_anomalies(df)
current = get_current()

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <div><span class="sc-header">SolarCast AI</span><span class="sc-badge">Live</span></div>
    <div class="sc-sub">Karnataka Renewable Energy Grid &nbsp;/&nbsp; Real-time Forecasting &nbsp;/&nbsp; Anomaly Detection</div>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Current Generation</div>
        <div class="kpi-value">{current['current_mw']}<span class="kpi-unit">MW</span></div>
        <div class="kpi-delta">Real-time feed</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Grid Capacity</div>
        <div class="kpi-value">{current['capacity_mw']}<span class="kpi-unit">MW</span></div>
        <div class="kpi-delta">Installed solar</div>
    </div>""", unsafe_allow_html=True)

with c3:
    delta_class = "warn" if current['status'] != 'normal' else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Utilization</div>
        <div class="kpi-value">{current['utilization_pct']}<span class="kpi-unit">%</span></div>
        <div class="kpi-delta {delta_class}">{'Normal operating range' if current['status'] == 'normal' else 'Below threshold'}</div>
    </div>""", unsafe_allow_html=True)

with c4:
    delta_class = "warn" if len(anomalies) > 0 else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Anomalies Detected</div>
        <div class="kpi-value">{len(anomalies)}<span class="kpi-unit">events</span></div>
        <div class="kpi-delta {delta_class}">{'Requires attention' if len(anomalies) > 0 else 'All systems normal'}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ── Main content ──────────────────────────────────────────────────────────────

col_left, col_right = st.columns([2.2, 1])

with col_left:
    st.markdown('<div class="section-title">72-Hour Generation Forecast</div>', unsafe_allow_html=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df['ds'], forecast_df['ds'][::-1]]),
        y=pd.concat([forecast_df['yhat_upper'], forecast_df['yhat_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(0, 200, 150, 0.07)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='#00c896', width=2),
        hovertemplate='%{x|%b %d %H:%M}<br>%{y:.0f} MW<extra></extra>'
    ))

    fig.add_hline(
        y=500, line_dash="dot", line_color="#1a2a4a",
        annotation_text="500 MW threshold",
        annotation_font_color="#3a5a8a",
        annotation_font_size=11
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#4a6a8a', family='sans-serif', size=12),
        xaxis=dict(gridcolor='#0d1a2e', linecolor='#1a2a4a', tickformat='%b %d\n%H:%M', title=None),
        yaxis=dict(gridcolor='#0d1a2e', linecolor='#1a2a4a', title='MW', title_font=dict(color='#3a5a8a', size=11)),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#4a6a8a')),
        height=340, margin=dict(l=0, r=0, t=10, b=0), hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    peak_row = forecast_df.loc[forecast_df['yhat'].idxmax()]
    m1, m2, m3 = st.columns(3)
    m1.metric("Peak Forecast", f"{peak_row['yhat']} MW")
    m2.metric("Peak Time", str(peak_row['ds']).split(" ")[1][:5])
    m3.metric("Avg Next 24h", f"{round(forecast_df.head(24)['yhat'].mean(), 1)} MW")

with col_right:
    st.markdown('<div class="section-title">Anomaly Alerts</div>', unsafe_allow_html=True)

    if anomalies:
        for a in anomalies[:7]:
            ts = str(a['timestamp'])
            time_str = ts.split(" ")[1][:5] if " " in ts else ts[:10]
            st.markdown(f"""
            <div class="alert-box">
                <div class="alert-title">Generation drop — {time_str}</div>
                <div class="alert-detail">
                    Actual: {a['actual_mw']} MW &nbsp;/&nbsp; Expected: {a['expected_mw']} MW<br>
                    Drop: <span style="color:#e05c5c;font-weight:600">{a['drop_pct']}%</span> below baseline
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#0a1f1a;border:1px solid #0d3d2e;border-radius:8px;padding:16px;text-align:center;">
            <div style="color:#00c896;font-size:13px;font-weight:600;">All systems nominal</div>
            <div style="color:#2a5a4a;font-size:12px;margin-top:4px;">No anomalies in monitoring window</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ── 24h bar chart ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-title">Hourly Breakdown — Next 24 Hours</div>', unsafe_allow_html=True)

df24 = forecast_df.head(24).copy()
df24['hour'] = df24['ds'].dt.strftime('%H:%M')
df24['color'] = df24['yhat'].apply(lambda x: '#00c896' if x > 500 else ('#f0a040' if x > 150 else '#c0392b'))

fig2 = go.Figure(go.Bar(
    x=df24['hour'], y=df24['yhat'].round(1),
    marker_color=df24['color'],
    text=df24['yhat'].round(0).astype(int),
    textposition='outside',
    textfont=dict(color='#4a6a8a', size=10),
    hovertemplate='%{x}<br>%{y:.0f} MW<extra></extra>'
))

fig2.update_layout(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#4a6a8a', size=11),
    xaxis=dict(gridcolor='#0d1a2e', linecolor='#1a2a4a'),
    yaxis=dict(gridcolor='#0d1a2e', linecolor='#1a2a4a', title='MW'),
    height=260, margin=dict(l=0, r=0, t=20, b=0),
    showlegend=False, bargap=0.3
)

st.plotly_chart(fig2, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="border-top:1px solid #0d1a2e;margin-top:1.5rem;padding-top:1rem;
     font-size:11px;color:#1a3a5a;display:flex;justify-content:space-between;">
    <span>SolarCast AI &nbsp;·&nbsp; Built for KREDL / KSPDCL &nbsp;·&nbsp; Prototype v1.0</span>
    <span>Prophet + LSTM &nbsp;·&nbsp; NASA POWER &nbsp;·&nbsp; FastAPI &nbsp;·&nbsp; Open-Meteo</span>
</div>
""", unsafe_allow_html=True)