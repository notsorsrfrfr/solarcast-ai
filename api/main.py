
import sys
import os
import math
import random
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model.forecast import load_and_prepare, train_model, generate_forecast, detect_anomalies

app = FastAPI(title="SolarCast AI", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading data and training model on startup...")
df = load_and_prepare()
model = train_model(df)
print("Model ready.")


@app.get("/")
def root():
    return {
        "status": "SolarCast AI is running",
        "version": "1.0",
        "endpoints": ["/forecast", "/anomalies", "/current", "/docs"]
    }


@app.get("/forecast")
def get_forecast(hours: int = 72):
    """Returns next N hours of solar generation forecast"""
    try:
        hours = min(max(hours, 1), 168)
        forecast = generate_forecast(model, hours=hours)
        return {
            "location": "Karnataka, India",
            "model": "Facebook Prophet",
            "hours": hours,
            "unit": "MW",
            "count": len(forecast),
            "forecast": forecast
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/anomalies")
def get_anomalies():
    """Returns recent anomaly events detected in generation data"""
    try:
        anomalies = detect_anomalies(df)
        return {
            "total_anomalies": len(anomalies),
            "threshold": "35% drop below expected",
            "anomalies": anomalies
        }
    except Exception as e:
        return {"error": str(e), "total_anomalies": 0, "anomalies": []}


@app.get("/current")
def get_current():
    """Returns simulated current generation stats"""
    try:
        hour = datetime.now().hour
        base = 850.0 * math.exp(-0.5 * ((hour - 13) / 3.5) ** 2)
        current = round(max(base * random.uniform(0.85, 1.1), 0.0), 1)
        capacity = 850.0
        utilization = round((current / capacity) * 100, 1)

        return {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M'),
            "current_mw": float(current),
            "capacity_mw": float(capacity),
            "utilization_pct": float(utilization),
            "status": "normal" if current > 100 else "low"
        }
    except Exception as e:
        return {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M'),
            "current_mw": 0.0,
            "capacity_mw": 850.0,
            "utilization_pct": 0.0,
            "status": "error",
            "detail": str(e)
        }


@app.get("/summary")
def get_summary():
    """Returns a high level grid summary for dashboard header"""
    try:
        hour = datetime.now().hour
        base = 850.0 * math.exp(-0.5 * ((hour - 13) / 3.5) ** 2)
        current = round(max(base * random.uniform(0.85, 1.1), 0.0), 1)
        anomalies = detect_anomalies(df)
        forecast = generate_forecast(model, hours=24)

        avg_24h = round(
            sum(f['yhat'] for f in forecast) / len(forecast), 1
        ) if forecast else 0.0

        peak = max(forecast, key=lambda x: x['yhat']) if forecast else {}

        return {
            "current_mw": float(current),
            "capacity_mw": 850.0,
            "utilization_pct": round((current / 850.0) * 100, 1),
            "status": "normal" if current > 100 else "low",
            "anomaly_count": len(anomalies),
            "avg_next_24h_mw": avg_24h,
            "peak_forecast_mw": float(round(peak.get('yhat', 0), 1)),
            "peak_forecast_time": peak.get('ds', 'N/A'),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M')
        }
    except Exception as e:
        return {"error": str(e)}