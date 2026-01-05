import requests
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("AQICN_API_KEY")
CITY = "Lahore"

def fetch_data():
    url = f"https://api.waqi.info/feed/{CITY}/?token={API_KEY}"
    response = requests.get(url).json()

    if response["status"] != "ok":
        raise Exception("API Error")

    aqi = response["data"]["aqi"]
    time = response["data"]["time"]["s"]

    df = pd.DataFrame([{
        "timestamp": pd.to_datetime(time),
        "aqi": aqi
    }])

    return df

if __name__ == "__main__":
    print(fetch_data())
