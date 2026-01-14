"""
Train Logistic Regression Model for AQI Prediction (Classification)
"""

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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = "aqi_db"
COLLECTION_NAME = "aqi_features"
CITY = "Islamabad"
MODEL_REGISTRY_DIR = Path("model_registry")
MODEL_REGISTRY_DIR.mkdir(exist_ok=True)

def fetch_training_data():
    """Fetch historical AQI data from MongoDB"""
    print("Fetching training data from MongoDB...")
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
        print(f"Error fetching data: {e}")
        raise
    finally:
        client.close()

def prepare_data(df):
    """Prepare features and target for training"""
    print("\nPreparing data...")
    
    features = ["hour", "day", "month", "day_of_week", "is_weekend", 
                "pm2_5", "pm10", "temperature", "humidity",
                "aqi_change", "aqi_3h_avg", "aqi_12h_avg", "pm_ratio"]
    target = "aqi"
    
    df_clean = df[features + [target]].dropna()
    
    print(f"Records after cleaning: {len(df_clean)}")
    print(f"Features: {len(features)}")
    
    if len(df_clean) < 50:
        raise ValueError(f"Not enough data. Need ≥50, got {len(df_clean)}")
    
    X = df_clean[features]
    y = df_clean[target]
    
    # Convert AQI to classes (1-5 levels)
    y_classes = pd.cut(y, bins=[0, 1, 2, 3, 4, 5], labels=[1, 2, 3, 4, 5], include_lowest=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_classes, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {len(X_train)} samples")
    print(f" Test set: {len(X_test)} samples")
    print(f"AQI Classes: 1 (Excellent) to 5 (Very Hazardous)")
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': features
    }

def train_logistic(data):
    """Train Logistic Regression model"""
    print("\nTraining Logistic Regression (Classification)...")
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(data['X_train'], data['y_train'])
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model"""
    y_pred = model.predict(X_test)
    y_pred_numeric = pd.to_numeric(y_pred, errors='coerce')
    y_test_numeric = pd.to_numeric(y_test, errors='coerce')
    
    rmse = np.sqrt(mean_squared_error(y_test_numeric, y_pred_numeric))
    mae = mean_absolute_error(y_test_numeric, y_pred_numeric)
    r2 = r2_score(y_test_numeric, y_pred_numeric)
    
    print(f"\n{'='*50}")
    print(f"Logistic Regression Performance")
    print(f"{'='*50}")
    print(f"  R² Score:   {r2:.4f}")
    print(f"  RMSE:       {rmse:.4f}")
    print(f"  MAE:        {mae:.4f}")
    print(f"{'='*50}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def save_to_registry(model, scaler, data, metrics):
    """Save model to registry"""
    print("\nSaving to Model Registry...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = MODEL_REGISTRY_DIR / f"logistic_{timestamp}"
    model_dir.mkdir(exist_ok=True)
    
    # Save model
    joblib.dump(model, model_dir / "model.pkl")
    joblib.dump(scaler, model_dir / "scaler.pkl")
    
    # Save metadata
    metadata = {
        'model_name': 'Logistic Regression',
        'type': 'Classification (AQI Levels 1-5)',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'r2': float(metrics['r2']),
            'rmse': float(metrics['rmse']),
            'mae': float(metrics['mae'])
        },
        'data_info': {
            'features': data['feature_names'],
            'n_training_samples': len(data['X_train']),
            'n_test_samples': len(data['X_test']),
            'city': CITY,
            'classes': {
                '1': 'Excellent',
                '2': 'Good',
                '3': 'Moderate',
                '4': 'Poor',
                '5': 'Very Hazardous'
            }
        }
    }
    
    with open(model_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    with open(model_dir / "features.json", 'w') as f:
        json.dump({'features': data['feature_names']}, f, indent=2)
    
    print(f"Saved to: {model_dir}")
    return model_dir

def main():
    print("\n" + "="*50)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("="*50)
    
    try:
        df = fetch_training_data()
        if len(df) == 0:
            print(" No data available!")
            return
        
        data = prepare_data(df)
        model = train_logistic(data)
        metrics = evaluate_model(model, data['X_test'], data['y_test'])
        model_dir = save_to_registry(model, data['scaler'], data, metrics)
        
        print("\n" + "="*50)
        print("LOGISTIC REGRESSION TRAINING COMPLETE!")
        print("="*50)
        print(f"Saved to: {model_dir}")
        print(f"R² Score: {metrics['r2']:.4f}")
        print("="*50)
        
    except Exception as e:
        print(f"\n Error: {e}")
        raise

if __name__ == "__main__":
    main()
