import pandas as pd
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import json
import joblib
from datetime import datetime
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = "aqi_db"
FEATURE_COLLECTION = "aqi_features"
MODEL_REGISTRY_COLLECTION = "model_registry"

CITY = "Islamabad"

MODEL_REGISTRY_DIR = Path("model_registry")
MODEL_REGISTRY_DIR.mkdir(exist_ok=True)

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

def prepare_data(df):
    features = [
        "hour", "day", "month", "day_of_week", "is_weekend",
        "pm2_5", "pm10", "temperature", "humidity",
        "aqi_change", "aqi_3h_avg", "aqi_12h_avg", "pm_ratio"
    ]
    target = "aqi_class"

    # Use AQI values 0-5 directly as classes
    df["aqi_class"] = df["aqi"].astype(int)

    # Check class distribution
    print("Class distribution:\n", df["aqi_class"].value_counts())

    df_clean = df[features + [target]].dropna()

    if len(df_clean) < 50:
        raise ValueError("Not enough data for training")

    X = df_clean[features]
    y = df_clean[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
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

def train_logistic(data):
    # Multi-class logistic regression with imbalanced classes handled
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=500,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(data["X_train"], data["y_train"])
    return model

def evaluate_model(model, X_test, y_test):
    # Predict class labels
    preds = model.predict(X_test)

    # Regression-style metrics using numeric class labels
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist()
    }

def save_to_registry(model, scaler, data, metrics):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = MODEL_REGISTRY_DIR / f"logistic_multiclass_{timestamp}"
    model_dir.mkdir()

    # Save model artifacts
    joblib.dump(model, model_dir / "model.pkl")
    joblib.dump(scaler, model_dir / "scaler.pkl")

    # Save metadata locally
    metadata = {
        "model_name": "LogisticRegression_MultiClass",
        "city": CITY,
        "trained_at": datetime.now().isoformat(),
        "metrics": metrics,
        "features": data["feature_names"],
        "n_training_samples": data["n_train"],
        "n_test_samples": data["n_test"],
        "model_path": str(model_dir)
    }

    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save metadata to MongoDB
    client = MongoClient(MONGODB_URI)
    try:
        db = client[DB_NAME]
        registry = db[MODEL_REGISTRY_COLLECTION]
        registry.insert_one(metadata)
        print("Model metadata saved to MongoDB registry")
    finally:
        client.close()

    return model_dir, metadata

def main():
    df = fetch_training_data()
    if df.empty:
        print("No data found. Run feature pipeline first.")
        return

    data = prepare_data(df)
    model = train_logistic(data)
    metrics = evaluate_model(model, data["X_test"], data["y_test"])
    model_dir, meta = save_to_registry(model, data["scaler"], data, metrics)

    print("\nTraining complete")
    print(f"Model saved at: {model_dir}")
    print(f"Metrics: {metrics}")

if __name__ == "__main__":
    main()
