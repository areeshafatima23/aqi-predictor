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
CITY_LAT = 33.6844  # Islamabad coordinates
CITY_LNG = 74.3131

def fetch_aqi():    
    try:
        url = "https://api.openweathermap.org/data/2.5/air_pollution"
        params = {
            "lat": CITY_LAT,
            "lon": CITY_LNG,
            "appid": OPENWEATHER_API_KEY
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract pollution data
        if not data.get("list"):
            raise Exception("No pollution data returned from OpenWeatherMap")
        
        latest = data["list"][0]  # Get most recent reading
        components = latest.get("components", {})
        main = latest.get("main", {})
        
        row = {
            "city": CITY,
            "timestamp": datetime.fromtimestamp(latest["dt"]),
            "aqi": main.get("aqi", 0),  # AQI level (1-5)
            "pm2_5": components.get("pm2_5", 0),
            "pm10": components.get("pm10", 0),
            "temperature": 20,  # OpenWeatherMap pollution API doesn't return temp/humidity
            "humidity": 50  # Use defaults or fetch from weather API separately
        }
        
        return pd.DataFrame([row])
    
    except Exception as e:
        print(f"Error fetching pollution data: {e}")
        raise
    
# Data Processing and Feature Engineering
def build_features(df):
    """Compute time-based and derived features from raw data"""
    # Handle missing values first - fill with 0 for numeric columns
    numeric_cols = ["pm2_5", "pm10", "temperature", "humidity"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Cast to correct types to match Feature Group schema (bigint = int64)
    df["pm2_5"] = df["pm2_5"].astype("int64")  # bigint
    df["pm10"] = df["pm10"].astype("int64")   # bigint
    df["temperature"] = df["temperature"].astype("int64")  # bigint
    df["humidity"] = df["humidity"].astype("float64")  # double
    df["aqi"] = df["aqi"].astype("int64")  # bigint
    
    # Time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
    # Derived features - rolling statistics for trend detection
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # AQI change rate (hour-over-hour)
    df["aqi_change"] = df["aqi"].diff().fillna(0).astype("float64")
    
    # Rolling averages (3-hour and 12-hour)
    df["aqi_3h_avg"] = df["aqi"].rolling(window=3, min_periods=1).mean().astype("float64")
    df["aqi_12h_avg"] = df["aqi"].rolling(window=12, min_periods=1).mean().astype("float64")
    
    # Pollutant ratios - handle zero division
    df["pm_ratio"] = (df["pm2_5"] / (df["pm10"] + 0.001)).fillna(0).astype("float64")
    df["pm_ratio"] = df["pm_ratio"].replace([float('inf'), float('-inf')], 0)
    
    # Temperature bands for seasonal patterns
    df["temp_band"] = pd.cut(df["temperature"], bins=[-50, 0, 15, 25, 35, 60], 
                             labels=["cold", "cool", "mild", "warm", "hot"]).astype(str)
    
    # Ensure no null values remain
    df = df.fillna(0)
    
    return df

def store_features(df):
    print("Connecting to MongoDB...")
    try:
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Convert dataframe to list of dictionaries
        records = df.to_dict(orient='records')
        
        print(f"Inserting {len(records)} rows into MongoDB...")
        print("Data to insert:")
        print(df)
        
        # Insert records
        result = collection.insert_many(records)
        print(f"Data inserted successfully! Inserted IDs: {result.inserted_ids[:5]}...")
        
    except Exception as e:
        print(f"Insert failed: {e}")
        raise
    finally:
        client.close()

if __name__ == "__main__":
    df = fetch_aqi()
    df = build_features(df)
    store_features(df)

    print("AQI data stored successfully")
