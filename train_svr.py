import pandas as pd
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import json
import pickle
import joblib
from datetime import datetime
from bson import Binary
from pathlib import Path

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

load_dotenv()

# ===============================
# CONFIG
# ===============================
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = "aqi_db"
FEATURE_COLLECTION = "aqi_features"
MODEL_REGISTRY_COLLECTION = "model_registry"

CITY = "Islamabad"

MODEL_REGISTRY_DIR = Path("model_registry")
MODEL_REGISTRY_DIR.mkdir(exist_ok=True)


# ===============================
# FETCH DATA
# ===============================
def fetch_training_data():
    print("Fetching training data from MongoDB...")
    client = MongoClient(MONGODB_URI)
    try:
        db = client[DB_NAME]
        collection = db[FEATURE_COLLECTION]

        records = list(collection.find())
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df.drop(columns=["_id"], inplace=True, errors="ignore")
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        print(f"Retrieved {len(df)} records")
        return df

    finally:
        client.close()


# ===============================
# PREPARE DATA
# ===============================
def prepare_data(df):
    # ❗ pm2_5 REMOVED from features
    features = [
        "hour", "day", "month", "day_of_week", "is_weekend",
        "pm10", "temperature", "humidity",
        "aqi_change", "aqi_3h_avg", "aqi_12h_avg", "pm_ratio"
    ]

    # ✅ PM2.5 is the regression target
    target = "pm2_5"

    df_clean = df[features + [target]].dropna()

    if len(df_clean) < 50:
        raise ValueError("Not enough data for training")

    X = df_clean[features]
    y = df_clean[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": features,
        "n_train": len(X_train),
        "n_test": len(X_test)
    }


# ===============================
# TRAIN MODEL
# ===============================
def train_svr(data):
    model = SVR(
        kernel="rbf",
        C=100,
        gamma="scale",
        epsilon=0.1
    )
    model.fit(data["X_train"], data["y_train"])
    return model


# ===============================
# EVALUATION
# ===============================
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds))
    }


# ===============================
# SAVE TO REGISTRY
# ===============================
def save_to_registry(model, scaler, data, metrics):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = MODEL_REGISTRY_DIR / f"svr_{timestamp}"
    model_dir.mkdir()

    joblib.dump(model, model_dir / "model.pkl")
    joblib.dump(scaler, model_dir / "scaler.pkl")

    # Serialize model & scaler as binary for MongoDB storage
    model_binary = Binary(pickle.dumps(model))
    scaler_binary = Binary(pickle.dumps(scaler))

    metadata = {
        "model_name": "SVR",
        "city": CITY,
        "trained_at": datetime.now().isoformat(),
        "metrics": metrics,
        "features": data["feature_names"],
        "target": "pm2_5",
        "n_training_samples": data["n_train"],
        "n_test_samples": data["n_test"],
        "model_path": str(model_dir),
        "model_binary": model_binary,
        "scaler_binary": scaler_binary
    }

    # Save metadata (without binaries) to local JSON
    meta_local = {k: v for k, v in metadata.items() if k not in ("model_binary", "scaler_binary")}
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(meta_local, f, indent=2)

    client = MongoClient(MONGODB_URI)
    try:
        db = client[DB_NAME]
        registry = db[MODEL_REGISTRY_COLLECTION]
        registry.insert_one(metadata)
        print("Model metadata + binaries saved to MongoDB registry")
    finally:
        client.close()

    return model_dir, metadata


# ===============================
# MAIN
# ===============================
def main():
    df = fetch_training_data()
    if df.empty:
        print("No data found. Run feature pipeline first.")
        return

    data = prepare_data(df)
    model = train_svr(data)
    metrics = evaluate_model(model, data["X_test"], data["y_test"])
    model_dir, meta = save_to_registry(model, data["scaler"], data, metrics)

    print("\nTraining complete")
    print(f"Model saved at: {model_dir}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
