import os
import pickle
import joblib
import pandas as pd
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

# BASE_DIR points to the repo root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# MongoDB Atlas URI and database/collections
MONGODB_URI = os.getenv("MONGODB_URI")  
DB_NAME = os.getenv("DB_NAME", "aqi_db")
FEATURE_COLLECTION = os.getenv("FEATURE_COLLECTION", "aqi_features")
MODEL_REGISTRY_COLLECTION = os.getenv("MODEL_REGISTRY_COLLECTION", "model_registry")

def pm25_to_aqi(pm25):
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]
    for cl, ch, il, ih in breakpoints:
        if cl <= pm25 <= ch:
            return ((pm25 - cl) / (ch - cl)) * (ih - il) + il
    return 500


def aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy (Sensitive)"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"

def get_best_model():
    client = MongoClient(MONGODB_URI)
    try:
        registry = client[DB_NAME][MODEL_REGISTRY_COLLECTION]

        # Today's date range (midnight to midnight)
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)
        today_str_start = today_start.isoformat()
        today_str_end = today_end.isoformat()

        # --- Priority 1: best model trained TODAY with binary data ---
        best = registry.find_one(
            {
                "target": "pm2_5",
                "model_binary": {"$exists": True},
                "trained_at": {"$gte": today_str_start, "$lt": today_str_end}
            },
            sort=[("metrics.rmse", 1)]
        )

        if best:
            model = pickle.loads(best["model_binary"])
            scaler = pickle.loads(best["scaler_binary"])
            print(f"Loaded TODAY's best model '{best['model_name']}' (RMSE={best['metrics']['rmse']:.4f}) from MongoDB binary")
            return model, scaler, best

        # --- Priority 2: best model with binary data (any day) ---
        best = registry.find_one(
            {"target": "pm2_5", "model_binary": {"$exists": True}},
            sort=[("metrics.rmse", 1)]
        )

        if best:
            model = pickle.loads(best["model_binary"])
            scaler = pickle.loads(best["scaler_binary"])
            print(f"No model trained today; loaded best available '{best['model_name']}' (RMSE={best['metrics']['rmse']:.4f}) from MongoDB binary")
            return model, scaler, best

        # --- Fallback: old entries without binary (local path) ---
        best = registry.find_one(
            {"target": "pm2_5"},
            sort=[("metrics.rmse", 1)]
        )
        if not best:
            raise RuntimeError("No PM2.5 model found in registry")

        relative_model_path = best["model_path"].replace("\\", "/")
        absolute_model_path = os.path.join(BASE_DIR, relative_model_path)
        model = joblib.load(os.path.join(absolute_model_path, "model.pkl"))
        scaler = joblib.load(os.path.join(absolute_model_path, "scaler.pkl"))
        print(f"Loaded model '{best['model_name']}' from local path (no binary in DB)")

        return model, scaler, best
    finally:
        client.close()

def fetch_latest_features(feature_names, window=24):
    client = MongoClient(MONGODB_URI)
    try:
        collection = client[DB_NAME][FEATURE_COLLECTION]
        df = pd.DataFrame(list(collection.find().sort("timestamp", -1).limit(window)))

        if df.empty:
            raise RuntimeError("No feature data found for inference")

        df.drop(columns=["_id"], inplace=True, errors="ignore")
        return df[feature_names]
    finally:
        client.close()

def run_inference():
    model, scaler, meta = get_best_model()
    feature_names = meta["features"]

    X = fetch_latest_features(feature_names)
    X_scaled = scaler.transform(X)

    pm25_preds = model.predict(X_scaled)
    pm25_preds = np.clip(pm25_preds, 0.1, 500)

    recent_pm25 = np.mean(pm25_preds[-8:])  # last 8 hours avg
    today = datetime.now().date()

    forecast = []
    for i in range(3):
        pm25_day = recent_pm25 * (1 + 0.02 * i)
        pm25_day = round(float(pm25_day), 2)
        aqi = round(pm25_to_aqi(pm25_day), 2)

        forecast.append({
            "date": str(today + timedelta(days=i + 1)),
            "pm2_5": pm25_day,
            "aqi": aqi,
            "category": aqi_category(aqi)
        })

    return forecast, meta
