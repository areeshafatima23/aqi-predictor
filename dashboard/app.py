import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="AQI Forecast", layout="wide")

st.markdown(
    "<h1 style='text-align:center;'>AQI Forecast Dashboard – Islamabad</h1>",
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    fetch = st.button("Get Latest Forecast", use_container_width=True)

# AQI CATEGORY COLORS
AQI_COLORS = {
    "Good": "#00e400",
    "Moderate": "#ffff00",
    "Unhealthy (Sensitive)": "#ff7e00",
    "Unhealthy": "#ff0000",
    "Very Unhealthy": "#8f3f97",
    "Hazardous": "#7e0023"
}

if fetch:
    with st.spinner("Loading predictions..."):
        response = requests.get(API_URL)

    if response.status_code != 200:
        st.error("Prediction API is not reachable.")
        st.stop()

    data = response.json()
    forecast = pd.DataFrame(data["forecast"])
    forecast["date"] = pd.to_datetime(forecast["date"])
    forecast["day"] = forecast["date"].dt.day_name()
    forecast["date_str"] = forecast["date"].dt.strftime("%d %b %Y")

    st.markdown(
        f"""
        <div style="text-align:center; margin-top:10px;">
            <b>Model Used:</b> {data['model_used']} &nbsp; | &nbsp;
            <b>RMSE:</b> {round(data['rmse'], 3)}
            <br>
            <span style="font-size:13px;">
                Model selected automatically based on lowest validation error
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("3-Day AQI Forecast")

    cols = st.columns(3)
    for i, row in forecast.iterrows():
        with cols[i]:
            st.markdown(
                f"""
                <div style="
                    background-color:{AQI_COLORS[row['category']]};
                    padding:20px;
                    border-radius:14px;
                    text-align:center;
                    color:black;
                ">
                    <h4>{row['day']}, {row['date_str']}</h4>
                    <h2>{row['aqi']} AQI</h2>
                    <p>PM2.5: <b>{row['pm2_5']} µg/m³</b></p>
                    <b>{row['category']}</b>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    hourly_blocks = []

    for _, row in forecast.iterrows():
        hours = pd.date_range(start=row["date"], periods=24, freq="H")
        hourly_blocks.append(
            pd.DataFrame({
                "datetime": hours,
                "aqi": np.full(24, row["aqi"])
            })
        )

    hourly_df = pd.concat(hourly_blocks).reset_index(drop=True)
    hourly_df["aqi_smooth"] = hourly_df["aqi"].rolling(
        window=6, center=True, min_periods=1
    ).mean()
    fig = go.Figure()

    fig.add_hrect(y0=0, y1=50, fillcolor="#00e400", opacity=0.15, line_width=0)
    fig.add_hrect(y0=50, y1=100, fillcolor="#ffff00", opacity=0.15, line_width=0)
    fig.add_hrect(y0=100, y1=150, fillcolor="#ff7e00", opacity=0.15, line_width=0)
    fig.add_hrect(y0=150, y1=200, fillcolor="#ff0000", opacity=0.15, line_width=0)
    fig.add_hrect(y0=200, y1=300, fillcolor="#8f3f97", opacity=0.15, line_width=0)
    fig.add_hrect(y0=300, y1=500, fillcolor="#7e0023", opacity=0.15, line_width=0)

    fig.add_trace(
        go.Scatter(
            x=hourly_df["datetime"],
            y=hourly_df["aqi_smooth"],
            mode="lines+markers",
            line=dict(color="red", width=3),
            marker=dict(size=6),
            name="Predicted AQI",
            hovertemplate=
                "<b>%{x|%a, %d %b %Y %H:%M}</b><br>"
                "AQI: %{y:.1f}<extra></extra>"
        )
    )

    fig.update_layout(
        title="3 Day AQI Trend",
        xaxis_title="Time",
        yaxis_title="AQI",
        yaxis=dict(range=[0, 500]),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### AQI Categories")

    legend_cols = st.columns(6)
    for col, (label, color) in zip(legend_cols, AQI_COLORS.items()):
        with col:
            st.markdown(
                f"""
                <div style="
                    background-color:{color};
                    padding:10px;
                    border-radius:8px;
                    text-align:center;
                    font-size:13px;
                    color:black;
                ">
                    {label}
                </div>
                """,
                unsafe_allow_html=True
            )
