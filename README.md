# AQI Predictor 

A Machine Learning–based **Air Quality Index (AQI) prediction system** built using Python and deployed with **Streamlit Community Cloud**.  
The app predicts AQI values, displays pollution trends, and categorizes air quality levels in a user-friendly dashboard.

 **Live App:**  
 https://aqi-predictor-hz7du3d8rxzmvtlak77e5y.streamlit.app/

## Features

- AQI prediction using trained ML models
- 3-day AQI forecast visualization
- Deployed directly on **Streamlit Community Cloud**
- Feature Pipeline: Fetches AQI data, computes time-based features (hour, day, month), AQI change rate
- Training Pipeline: Supports Scikit-learn (Random Forest, Ridge Regression) and Deep Learning models
- Automated Pipeline: GitHub Actions for hourly feature updates and daily model retraining
- Web Dashboard: Interactive UI showing predictions and forecasts
- Explainability: SHAP/LIME for feature importance

## Machine Learning Pipeline

1. Data ingestion & feature engineering
2. Model training (Random Forest, Ridge, SVR)
3. Model evaluation using RMSE
4. Automatic best-model selection
5. Model registry versioning
6. Inference & AQI conversion
7. Visualization via Streamlit

## Project Structure

```text
aqi-predictor/
│
├── dashboard/          # Streamlit dashboard UI
├── inference/          # Model loading & prediction logic
├── model_registry/     # Versioned trained models & scalers
├── shap_analysis/      # Model explainability using SHAP
├── .github/workflows/  # CI pipelines for model training
├── train_svr.py          # Model training scripts
├── train_random_forest.py          # Model training scripts
├── train_ridge.py          # Model training scripts
├── pipeline.py         # Feature pipeline
├── requirements.txt    # Python dependencies
└── README.md
```
## How to Run Locally 

- Create & Activate Virtual Environment

```python -m venv venv ```


- Activate the environment

macOS / Linux ```source venv/bin/activate```


Windows ```venv\Scripts\activate```

- Install Dependencies
```pip install -r requirements.txt```

- Run Streamlit App
```streamlit run dashboard/app.py```

## Author
Areesha Fatima
