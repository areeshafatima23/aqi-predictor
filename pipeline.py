import requests
import pandas as pd
from dotenv import load_dotenv
import os
from pymongo import MongoClient
from datetime import datetime

# Load environment variables
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")

DB_NAME = "aqi_db"
COLLECTION_NAME = "aqi_features"

CITY = "Islamabad"
CITY_LAT = 33.6844
CITY_LNG = 74.3131

def fetch_current_weather():
    """Fetch current weather from Open-Meteo (no API key needed)"""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": CITY_LAT,
            "longitude": CITY_LNG,
            "current_weather": True
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        weather = data.get("current_weather", {})

        df = pd.DataFrame([{
            "timestamp": pd.to_datetime(weather.get("time")),
            "temperature": weather.get("temperature", 0),
            "humidity": weather.get("windspeed", 0)  # Open-Meteo doesn’t provide humidity directly; can adjust if needed
        }])
        return df

    except Exception as e:
        print(f"Error fetching weather: {e}")
        return pd.DataFrame()

def fetch_current_aqi():
    """Fetch current AQI from OpenWeatherMap"""
    try:
        url = "https://api.openweathermap.org/data/2.5/air_pollution"
        params = {
            "lat": CITY_LAT,
            "lon": CITY_LNG,
            "appid": OPENWEATHER_API_KEY
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        if not data.get("list"):
            raise Exception("No AQI data returned")

        pollution = data["list"][0]
        components = pollution.get("components", {})
        main = pollution.get("main", {})

        df = pd.DataFrame([{
            "city": CITY,
            "timestamp": datetime.fromtimestamp(pollution["dt"]),
            "aqi": main.get("aqi", 0),
            "pm2_5": components.get("pm2_5", 0),
            "pm10": components.get("pm10", 0)
        }])
        return df

    except Exception as e:
        print(f"Error fetching AQI: {e}")
        return pd.DataFrame()

def build_features(df):
    numeric_cols = ["pm2_5", "pm10", "temperature", "humidity"]
    for col in numeric_cols:
        df[col] = df[col].fillna(0)

    df["pm2_5"] = df["pm2_5"].astype("int64")
    df["pm10"] = df["pm10"].astype("int64")
    df["temperature"] = df["temperature"].astype("int64")
    df["humidity"] = df["humidity"].astype("float64")
    df["aqi"] = df["aqi"].astype("int64")

    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["aqi_change"] = df["aqi"].diff().fillna(0)
    df["aqi_3h_avg"] = df["aqi"].rolling(3, min_periods=1).mean()
    df["aqi_12h_avg"] = df["aqi"].rolling(12, min_periods=1).mean()
    df["pm_ratio"] = df["pm2_5"] / (df["pm10"] + 0.001)

    df["temp_band"] = pd.cut(
        df["temperature"],
        bins=[-50, 0, 15, 25, 35, 60],
        labels=["cold", "cool", "mild", "warm", "hot"]
    ).astype(str)

    return df.fillna(0)

def store_features(df):
    print("Connecting to MongoDB...")
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    records = df.to_dict(orient="records")
    collection.insert_many(records)
    print(f"Inserted {len(records)} new records")
    client.close()

if __name__ == "__main__":
    df_aqi = fetch_current_aqi()
    df_weather = fetch_current_weather()

    if df_aqi.empty or df_weather.empty:
        print("Data fetch failed, aborting pipeline.")
    else:
        df = pd.merge_asof(
            df_aqi.sort_values("timestamp"),
            df_weather.sort_values("timestamp"),
            on="timestamp",
            direction="nearest"
        )
        df = build_features(df)
        store_features(df)
        print("AQI & Weather data pipeline completed")
