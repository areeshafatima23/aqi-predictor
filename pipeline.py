import requests
import pandas as pd
from dotenv import load_dotenv
import os
from pymongo import MongoClient
from datetime import datetime

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")

DB_NAME = "aqi_db"
COLLECTION_NAME = "aqi_features"

CITY = "Islamabad"
CITY_LAT = 33.6844
CITY_LNG = 74.3131

def fetch_aqi_and_weather():
    try:
        # AQI API
        pollution_url = "https://api.openweathermap.org/data/2.5/air_pollution"
        pollution_params = {
            "lat": CITY_LAT,
            "lon": CITY_LNG,
            "appid": OPENWEATHER_API_KEY
        }

        pollution_response = requests.get(pollution_url, params=pollution_params)
        pollution_response.raise_for_status()
        pollution_data = pollution_response.json()

        if not pollution_data.get("list"):
            raise Exception("No pollution data returned")

        pollution = pollution_data["list"][0]
        components = pollution.get("components", {})
        main = pollution.get("main", {})

        # Weather API
        weather_url = "https://api.openweathermap.org/data/2.5/weather"
        weather_params = {
            "lat": CITY_LAT,
            "lon": CITY_LNG,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }

        weather_response = requests.get(weather_url, params=weather_params)
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        weather_main = weather_data.get("main", {})

        row = {
            "city": CITY,
            "timestamp": datetime.fromtimestamp(pollution["dt"]),

            # Pollution
            "aqi": main.get("aqi", 0),
            "pm2_5": components.get("pm2_5", 0),
            "pm10": components.get("pm10", 0),

            # Weather
            "temperature": weather_main.get("temp", 0),
            "humidity": weather_main.get("humidity", 0)
        }

        return pd.DataFrame([row])

    except Exception as e:
        print(f"Error fetching AQI or weather data: {e}")
        raise

def build_features(df):
    numeric_cols = ["pm2_5", "pm10", "temperature", "humidity"]
    for col in numeric_cols:
        df[col] = df[col].fillna(0)

    df["pm2_5"] = df["pm2_5"].astype("int64")
    df["pm10"] = df["pm10"].astype("int64")
    df["temperature"] = df["temperature"].astype("int64")
    df["humidity"] = df["humidity"].astype("float64")
    df["aqi"] = df["aqi"].astype("int64")

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Derived features
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
    df = fetch_aqi_and_weather()    # fetch real data
    df = build_features(df)         # build features
    store_features(df)              # save to MongoDB

    print("AQI & Weather data pipeline completed")
