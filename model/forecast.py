import pandas as pd
import numpy as np
from prophet import Prophet
import json

def load_and_prepare():
    df = pd.read_csv("data/solar_history.csv")
    df['ds'] = pd.to_datetime(df['ds'])
    return df

def train_model(df):
    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        interval_width=0.90
    )
    m.fit(df)
    return m

def generate_forecast(model, hours=72):
    future = model.make_future_dataframe(periods=hours, freq='h')
    forecast = model.predict(future)

    # Only return future predictions, clip negatives (solar can't be negative)
    future_only = forecast.tail(hours)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    future_only['yhat'] = future_only['yhat'].clip(0).round(2)
    future_only['yhat_lower'] = future_only['yhat_lower'].clip(0).round(2)
    future_only['yhat_upper'] = future_only['yhat_upper'].clip(0).round(2)
    future_only['ds'] = future_only['ds'].dt.strftime('%Y-%m-%d %H:%M')

    return future_only.to_dict(orient='records')

def detect_anomalies(df, threshold=0.35):
    """Flag hours where generation drops >35% below expected for that hour"""
    df = df.copy()
    df['hour'] = pd.to_datetime(df['ds']).dt.hour
    hourly_mean = df.groupby('hour')['y'].mean()
    df['expected'] = df['hour'].map(hourly_mean)
    df['drop_pct'] = (df['expected'] - df['y']) / (df['expected'] + 1)
    anomalies = df[df['drop_pct'] > threshold].tail(20)

    return [{
        "timestamp": str(row['ds']),
        "actual_mw": round(row['y'], 2),
        "expected_mw": round(row['expected'], 2),
        "drop_pct": round(row['drop_pct'] * 100, 1)
    } for _, row in anomalies.iterrows()]

if __name__ == "__main__":
    print("Loading data...")
    df = load_and_prepare()
    print("Training Prophet model (takes ~30 seconds)...")
    model = train_model(df)
    print("Generating 72-hour forecast...")
    forecast = generate_forecast(model)
    anomalies = detect_anomalies(df)
    print(f"\nFirst 3 forecast points: {json.dumps(forecast[:3], indent=2)}")
    print(f"\nAnomalies found: {len(anomalies)}")