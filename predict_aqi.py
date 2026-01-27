import pandas as pd
from pymongo import MongoClient
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = "aqi_db"
COLLECTION_NAME = "aqi_features"

CITY = "Islamabad"

def get_historical_data():
    """Fetch historical AQI data from MongoDB"""
    print("Fetching historical data from MongoDB...")
    try:
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        records = list(collection.find())
        if not records:
            print("No records found in MongoDB")
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"Retrieved {len(df)} records")
        return df
    
    except Exception as e:
        print(f"Error fetching data from MongoDB: {e}")
        raise
    finally:
        client.close()

def train_ridge_model(df):
    """Train Ridge Regression model to predict AQI"""
    print("\nRidge Regression model")
    
    features = ["hour", "day_of_week", "is_weekend", "pm2_5", "pm10", "temperature", "humidity"]
    target = "aqi"
    
    df_clean = df[features + [target]].dropna()
    if len(df_clean) < 10:
        print("Not enough data to train (need at least 10 records)")
        return None, None
    
    X = df_clean[features]
    y = df_clean[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)
    
    train_score = model.score(X_scaled, y)
    print(f"R² score: {train_score:.3f}")
    
    joblib.dump(model, "aqi_ridge_model.pkl")
    joblib.dump(scaler, "aqi_ridge_scaler.pkl")
    print("Model and scaler saved!")
    
    return model, scaler

def aqi_label(aqi):
    """Return AQI category label based on numeric value"""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def predict_next_3_days(model, scaler, current_data):
    """Predict AQI value per day for next 3 days"""
    if model is None:
        print("Model not trained yet.")
        return None

    current_time = pd.to_datetime(current_data["timestamp"].iloc[-1])
    
    current_pm25 = current_data["pm2_5"].iloc[-1]
    current_pm10 = current_data["pm10"].iloc[-1]
    current_temp = current_data["temperature"].iloc[-1]
    current_humidity = current_data["humidity"].iloc[-1]

    hourly_predictions = []

    # Predict next 72 hours
    for i in range(1, 73):
        future_time = current_time + timedelta(hours=i)
        hour = future_time.hour
        day_of_week = future_time.dayofweek
        is_weekend = 1 if day_of_week in [5, 6] else 0
        
        features = np.array([[hour, day_of_week, is_weekend,
                              current_pm25, current_pm10,
                              current_temp, current_humidity]])
        
        features_scaled = scaler.transform(features)
        predicted_aqi = model.predict(features_scaled)[0]
        predicted_aqi = max(0, predicted_aqi)
        
        hourly_predictions.append({
            "date": future_time.date(),
            "aqi": predicted_aqi
        })

    # Aggregate hourly → daily AQI
    df_hourly = pd.DataFrame(hourly_predictions)
    df_daily = df_hourly.groupby("date")["aqi"].mean().reset_index()

    df_daily["predicted_aqi"] = df_daily["aqi"].round(2)
    df_daily["city"] = CITY

    # Add AQI label
    df_daily["aqi_label"] = df_daily["predicted_aqi"].apply(aqi_label)

    return df_daily[["city", "date", "predicted_aqi", "aqi_label"]]

def main():
    df = get_historical_data()
    if df.empty:
        print("No historical data. Run your backfill first!")
        return
    
    model, scaler = train_ridge_model(df)
    
    if model:
        predictions = predict_next_3_days(model, scaler, df)
        if predictions is not None:
            print("\nNext 3 Days AQI Predictions:")
            for _, row in predictions.iterrows():
                print(f"{row['date']} - Predicted AQI: {row['predicted_aqi']} ({row['aqi_label']})")

if __name__ == "__main__":
    main()
