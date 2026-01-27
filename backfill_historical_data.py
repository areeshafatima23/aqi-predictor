import requests
import pandas as pd
from dotenv import load_dotenv
import os
from pymongo import MongoClient
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")

DB_NAME = "aqi_db"
COLLECTION_NAME = "aqi_features"

CITY = "Islamabad"
CITY_LAT = 33.6844
CITY_LNG = 74.3131
BACKFILL_DAYS = 90

def fetch_historical_aqi(start_timestamp, end_timestamp):
    """Fetch historical AQI from OpenWeatherMap"""
    try:
        url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
        params = {
            "lat": CITY_LAT,
            "lon": CITY_LNG,
            "start": int(start_timestamp),
            "end": int(end_timestamp),
            "appid": OPENWEATHER_API_KEY
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        records = [
            {
                "timestamp": datetime.fromtimestamp(item["dt"]),
                "aqi": item["main"].get("aqi", 0),
                "pm2_5": item["components"].get("pm2_5", 0),
                "pm10": item["components"].get("pm10", 0),
            }
            for item in data.get("list", [])
        ]
        df = pd.DataFrame(records)
        print(f"Fetched {len(df)} AQI records")
        return df

    except Exception as e:
        print("Error fetching AQI:", e)
        return pd.DataFrame()

def fetch_historical_weather(start_date, end_date):
    """Fetch historical weather data using Open-Meteo Archive API"""
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": CITY_LAT,
            "longitude": CITY_LNG,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relativehumidity_2m"
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        # Build weather dataframe
        times = data["hourly"]["time"]
        temps = data["hourly"]["temperature_2m"]
        humidity = data["hourly"]["relativehumidity_2m"]

        df_weather = pd.DataFrame({
            "timestamp": pd.to_datetime(times),
            "temperature": temps,
            "humidity": humidity
        })
        print(f"Fetched {len(df_weather)} weather records")
        return df_weather

    except Exception as e:
        print("Error fetching weather:", e)
        return pd.DataFrame()

def build_features(df):
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

    df = df.sort_values("timestamp").reset_index(drop=True)

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
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    coll = db[COLLECTION_NAME]
    coll.insert_many(df.to_dict("records"))
    print(f"Stored {len(df)} records")
    client.close()

def main():
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=BACKFILL_DAYS)

    # Format dates
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")

    df_aqi = fetch_historical_aqi(start_dt.timestamp(), end_dt.timestamp())
    if df_aqi.empty:
        print("No AQI data found")
        return

    df_weather = fetch_historical_weather(start_date, end_date)
    if df_weather.empty:
        print("No weather data found")
        return

    # Merge on nearest timestamp
    df = pd.merge_asof(
        df_aqi.sort_values("timestamp"),
        df_weather.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
    )

    df["city"] = CITY
    df = build_features(df)
    store_features(df)
    print("Backfill complete!")

if __name__ == "__main__":
    main()
