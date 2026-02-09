# AQI Predictor 

A Machine Learning–based **Air Quality Index (AQI) prediction system** built using Python and deployed with **Streamlit Community Cloud**.  
The app predicts AQI values, displays pollution trends, and categorizes air quality levels in a user-friendly dashboard.

 **Live App:**  
 https://aqi-predictor-hz7du3d8rxzmvtlak77e5y.streamlit.app/

## Features

- AQI prediction using trained ML models
- 3-day AQI forecast visualization
- PM2.5 & AQI trend charts
- Deployed directly on **Streamlit Community Cloud**
- Feature Pipeline: Fetches AQI data, computes time-based features (hour, day, month), AQI change rate
- Training Pipeline: Supports Scikit-learn (Random Forest, Ridge Regression) and Deep Learning models
- Automated Pipeline: GitHub Actions for hourly feature updates and daily model retraining
- Web Dashboard: Interactive UI showing predictions and forecasts
- Explainability: SHAP/LIME for feature importance


## Project Structure
aqi-predictor/
│
├── .devcontainer/              # Dev container config 
├── .github/workflows/          # GitHub Actions (model training, CI)
├── .streamlit/                 # Streamlit Cloud config
│
├── dashboard/
│   └── app.py                  # Streamlit entry point
│
├── inference/
│   ├── __init__.py
│   └── inference.py            # Model loading + prediction logic
│
├── model_registry/             # Trained models
│
├── shap_analysis/              # SHAP explainability 
│
├── train_random_forest.py
├── train_ridge.py
├── train_svr.py                # Model training scripts
│
├── pipeline.py                 # Feature pipeline
├── predict_aqi.py              # Local prediction
├── backfill_historical_data.py
│
├── aqi_ridge_model.pkl        
├── aqi_ridge_scaler.pkl
│
├── requirements.txt


## How to Run Locally 
Create & Activate Virtual Environment (Optional)
      python -m venv venv
      source venv/bin/activate        # macOS/Linux
      venv\Scripts\activate           # Windows
Install Dependencies
      pip install -r requirements.txt

Run Streamlit App
      streamlit run dashboard/app.py

## Author
Areesha Fatima
