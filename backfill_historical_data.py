import requests
import pandas as pd
from dotenv import load_dotenv
import os
from pymongo import MongoClient
from datetime import datetime, timedelta
import time

# Load environment variables
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = "aqi_db"
COLLECTION_NAME = "aqi_features"

CITY = "Islamabad"
CITY_LAT = 33.6844  # Islamabad coordinates
CITY_LNG = 74.3131
BACKFILL_DAYS = 90


def fetch_historical_pollution_data(start_timestamp, end_timestamp):
   
    try:
        url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
        params = {
            "lat": CITY_LAT,
            "lon": CITY_LNG,
            "start": int(start_timestamp),
            "end": int(end_timestamp),
            "appid": OPENWEATHER_API_KEY
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Parse the response
        records = []
        for item in data.get("list", []):
            timestamp = datetime.fromtimestamp(item["dt"], tz=None)
            components = item.get("components", {})
            main = item.get("main", {})
            
            record = {
                "timestamp": timestamp,
                "aqi": main.get("aqi", 0),  # Air Quality Index (1-5)
                "pm2_5": components.get("pm2_5", 0),
                "pm10": components.get("pm10", 0),
                "temperature": item.get("temp", 0), 
                "humidity": item.get("humidity", 0), 
            }
            records.append(record)
        
        df_pollution = pd.DataFrame(records)
        print(f"Retrieved {len(df_pollution)} hourly pollution records from OpenWeatherMap")
        return df_pollution
    
    except Exception as e:
        print(f"Error fetching pollution data: {e}")
        raise

def generate_historical_data_with_real_pollution(days=90):

    # Calculate timestamps
    end_timestamp = datetime.now()
    start_timestamp = end_timestamp - timedelta(days=days)
    
    # Fetch real pollution data from OpenWeatherMap
    df_pollution = fetch_historical_pollution_data(start_timestamp.timestamp(), end_timestamp.timestamp())
    
    if df_pollution is None or len(df_pollution) == 0:
        raise Exception("Failed to fetch real pollution data from OpenWeatherMap. Check API key and ensure data is available.")
    
    # Add city column
    df_pollution["city"] = CITY
    
    # Handle missing weather data - fill with reasonable defaults or aggregate
    if df_pollution["temperature"].sum() == 0:
        print("Temperature data not available from OpenWeatherMap, using defaults")
        df_pollution["temperature"] = 25  # Default to 25°C
    
    if df_pollution["humidity"].sum() == 0:
        print("Humidity data not available from OpenWeatherMap, using defaults")
        df_pollution["humidity"] = 50  # Default to 50%
    
    # Reorder columns
    df_result = df_pollution[["city", "timestamp", "aqi", "pm2_5", "pm10", "temperature", "humidity"]]
    
    print(f"Retrieved {len(df_result)} records with real pollution data from OpenWeatherMap")
    return df_result

def build_features(df):
    """Apply feature engineering to historical data"""
    print("Building features...")
    
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
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
    # Derived features - rolling statistics for trend detection
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # AQI change rate
    df["aqi_change"] = df["aqi"].diff().fillna(0).astype("float64")
    
    # Rolling averages
    df["aqi_3h_avg"] = df["aqi"].rolling(window=3, min_periods=1).mean().astype("float64")
    df["aqi_12h_avg"] = df["aqi"].rolling(window=12, min_periods=1).mean().astype("float64")
    
    # Pollutant ratios
    df["pm_ratio"] = (df["pm2_5"] / (df["pm10"] + 0.001)).fillna(0).astype("float64")
    
    # Temperature bands
    df["temp_band"] = pd.cut(df["temperature"], bins=[-50, 0, 15, 25, 35, 60], 
                             labels=["cold", "cool", "mild", "warm", "hot"]).astype(str)
    
    return df

def store_historical_features(df):
    try:
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Convert dataframe to list of dictionaries
        records = df.to_dict(orient='records')
        
        print(f"Inserting {len(records)} records into MongoDB...")
        
        # Insert records
        result = collection.insert_many(records)
        
        print(f"Successfully stored {len(result.inserted_ids)} historical records!")
        
    except Exception as e:
        print(f"Error storing data: {e}")
        raise
    finally:
        client.close()

def main():
    """Main backfill workflow"""
    print("=" * 70)
    print("AQI HISTORICAL DATA BACKFILL")
    print("=" * 70)
    
    # Fetch real historical pollution data from OpenWeatherMap
    print(f"\nFetching real historical pollution data for {BACKFILL_DAYS} days...")
    df = generate_historical_data_with_real_pollution(days=BACKFILL_DAYS)
    
    if df is None or len(df) == 0:
        print("No data to backfill!")
        return
    
    print(f"Retrieved {len(df)} records")
    
    # Build features
    df = build_features(df)
    
    print("\nFeature Summary:")
    print(f"  - Records: {len(df)}")
    print(f"  - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  - Features: {', '.join(df.columns)}")
    print(f"  - Missing values:\n{df.isnull().sum()}")
    
    # Store in Feature Store
    print("\n" + "=" * 70)
    store_historical_features(df)
    
    print("\n" + "=" * 70)
    print("=" * 70)

if __name__ == "__main__":
    main()
