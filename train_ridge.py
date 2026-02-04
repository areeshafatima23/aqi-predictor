import pandas as pd
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import json
import joblib
from datetime import datetime
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = "aqi_db"
FEATURE_COLLECTION = "aqi_features"
MODEL_REGISTRY_COLLECTION = "model_registry"

CITY = "Islamabad"

MODEL_REGISTRY_DIR = Path("model_registry")
MODEL_REGISTRY_DIR.mkdir(exist_ok=True)

# ===============================
# DATA
# ===============================
def fetch_training_data():
    client = MongoClient(MONGODB_URI)
    try:
        df = pd.DataFrame(list(client[DB_NAME][FEATURE_COLLECTION].find()))
        df.drop(columns=["_id"], inplace=True, errors="ignore")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    finally:
        client.close()

# ===============================
# PREPARE DATA (PM2.5 TARGET)
# ===============================
def prepare_data(df):
    features = [
        "hour", "day", "month", "day_of_week", "is_weekend",
        "pm10", "temperature", "humidity", "pm_ratio"
    ]

    target = "pm2_5"

    df = df[features + [target]].dropna()

    X = df[features]
    y = df[target]  # 1D Series ✔

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features

# ===============================
# TRAIN
# ===============================
def train_ridge(X_train, y_train):
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model

# ===============================
# EVAL
# ===============================
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds))
    }

# ===============================
# SAVE
# ===============================
def save(model, scaler, features, metrics):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = MODEL_REGISTRY_DIR / f"ridge_pm25_{ts}"
    path.mkdir()

    joblib.dump(model, path / "model.pkl")
    joblib.dump(scaler, path / "scaler.pkl")

    meta = {
        "model_name": "RidgeRegression",
        "target": "pm2_5",
        "city": CITY,
        "trained_at": datetime.now().isoformat(),
        "metrics": metrics,
        "features": features,
        "model_path": str(path)
    }

    with open(path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    MongoClient(MONGODB_URI)[DB_NAME][MODEL_REGISTRY_COLLECTION].insert_one(meta)

# ===============================
# MAIN
# ===============================
def main():
    df = fetch_training_data()
    X_tr, X_te, y_tr, y_te, scaler, features = prepare_data(df)
    model = train_ridge(X_tr, y_tr)
    metrics = evaluate(model, X_te, y_te)
    save(model, scaler, features, metrics)

    print("Training done:", metrics)

if __name__ == "__main__":
    main()
