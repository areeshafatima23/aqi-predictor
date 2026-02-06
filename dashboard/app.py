import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="AQI Forecast", layout="wide")

st.markdown(
    """
    <h1 style="text-align:center;">
        AQI Forecast Dashboard – Islamabad
    </h1>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    fetch = st.button("Get Latest Forecast", use_container_width=True)

# AQI color mapping
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
        st.error("API not running. Start FastAPI first.")
        st.stop()

    data = response.json()
    forecast = pd.DataFrame(data["forecast"])

    forecast["date"] = pd.to_datetime(forecast["date"])
    forecast["day"] = forecast["date"].dt.day_name()
    forecast["date_str"] = forecast["date"].dt.strftime("%d %b %Y")

    st.subheader("Model Info")
    st.write(f"**Model Used:** {data['model_used']}")
    st.write(f"**RMSE:** {round(data['rmse'], 3)}")
    st.caption("Model is selected automatically based on lowest validation error.")

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
    st.subheader("AQI Trend Analysis")

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.axhspan(0, 50, color="#00e400", alpha=0.2, label="Good")
    ax.axhspan(50, 100, color="#ffff00", alpha=0.2, label="Moderate")
    ax.axhspan(100, 150, color="#ff7e00", alpha=0.2, label="Unhealthy (Sensitive)")
    ax.axhspan(150, 200, color="#ff0000", alpha=0.2, label="Unhealthy")
    ax.axhspan(200, 300, color="#8f3f97", alpha=0.2, label="Very Unhealthy")
    ax.axhspan(300, 500, color="#7e0023", alpha=0.2, label="Hazardous")

    ax.plot(
        forecast["date"],
        forecast["aqi"],
        color="black",
        linewidth=2,
        marker="o",
        label="AQI Forecast"
    )

    ax.set_xticks(forecast["date"])
    ax.set_xticklabels(
        [d.strftime("%a, %d %b") for d in forecast["date"]],
        rotation=30
    )
    ax.set_ylim(0, 500)
    ax.set_ylabel("AQI")
    ax.set_xlabel("Date")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")

    st.pyplot(fig)
