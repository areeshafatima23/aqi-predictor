import pandas as pd
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor
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
    print("📥 Fetching historical data from MongoDB...")
    try:
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Fetch all records
        records = list(collection.find())
        
        if not records:
            print("No records found in MongoDB")
            return pd.DataFrame()
        
        # Remove MongoDB's _id field and convert to DataFrame
        df = pd.DataFrame(records)
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"Retrieved {len(df)} records")
        return df
    
    except Exception as e:
        print(f"Error fetching data from MongoDB: {e}")
        raise
    finally:
        client.close()

def train_model(df):
    """Train a Random Forest model to predict AQI"""
    print("\nTraining AQI prediction model...")
    
    # Features for prediction
    features = ["hour", "day_of_week", "is_weekend", "pm2_5", "pm10", "temperature", "humidity"]
    target = "aqi"
    
    # Remove rows with missing values
    df_clean = df[features + [target]].dropna()
    
    if len(df_clean) < 10:
        print("Not enough data to train (need at least 10 records)")
        return None, None
    
    X = df_clean[features]
    y = df_clean[target]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_scaled, y)
    
    # Evaluate
    train_score = model.score(X_scaled, y)
    print(f"Model trained! R² score: {train_score:.3f}")
    
    # Save model
    joblib.dump(model, "aqi_model.pkl")
    joblib.dump(scaler, "aqi_scaler.pkl")
    print("💾 Model saved!")
    
    return model, scaler

def predict_next_3_days(model, scaler, current_data):
    """Predict AQI for next 3 days (every hour)"""
    if model is None:
        print("❌ Model not trained yet. Need more historical data.")
        return None
    
    print("\n🔮 Predicting AQI for next 72 hours...")
    
    predictions = []
    current_time = pd.to_datetime(current_data["timestamp"].iloc[0])
    
    # Get current conditions for feature context
    current_pm25 = current_data["pm2_5"].iloc[0]
    current_pm10 = current_data["pm10"].iloc[0]
    current_temp = current_data["temperature"].iloc[0]
    current_humidity = current_data["humidity"].iloc[0]
    
    # Predict for next 72 hours
    for i in range(1, 73):  # 72 hours
        future_time = current_time + timedelta(hours=i)
        
        hour = future_time.hour
        day_of_week = future_time.dayofweek
        is_weekend = 1 if day_of_week in [5, 6] else 0
        
        # Use current conditions (in real scenario, you'd use weather forecast)
        features = np.array([[hour, day_of_week, is_weekend, current_pm25, current_pm10, current_temp, current_humidity]])
        features_scaled = scaler.transform(features)
        
        predicted_aqi = model.predict(features_scaled)[0]
        predicted_aqi = max(0, predicted_aqi)  # AQI can't be negative
        
        predictions.append({
            "city": CITY,
            "timestamp": future_time,
            "predicted_aqi": round(predicted_aqi, 2),
            "hour": hour,
            "day_of_week": day_of_week
        })
    
    return pd.DataFrame(predictions)

def main():
    try:
        # Get historical data
        df = get_historical_data()
        
        if len(df) == 0:
            print("❌ No historical data available yet. Run pipeline.py first!")
            return
        
        # Train model
        model, scaler = train_model(df)
        
        # Predict next 3 days
        if model:
            predictions = predict_next_3_days(model, scaler, df)
            
            if predictions is not None:
                print("\n📊 Next 3 Days AQI Predictions:")
                print("=" * 70)
                for idx, row in predictions.iterrows():
                    if idx % 24 == 0:  # Show every day at midnight
                        print(f"\n{row['timestamp'].strftime('%Y-%m-%d %H:%M')} - Predicted AQI: {row['predicted_aqi']}")
                    elif idx % 6 == 0:  # Also show every 6 hours
                        print(f"{row['timestamp'].strftime('%H:%M')} - Predicted AQI: {row['predicted_aqi']}")
                
                print("\n💾 Saving predictions to CSV...")
                predictions.to_csv("aqi_predictions_3days.csv", index=False)
                print("✅ Predictions saved to aqi_predictions_3days.csv")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
