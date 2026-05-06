import pandas as pd
import numpy as np

def generate_solar_data(days=365):
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days * 24, freq='h')
    df = pd.DataFrame({'ds': dates})

    # Realistic Karnataka solar pattern:
    # Peak generation ~1pm, zero at night, seasonal variation
    hour = df['ds'].dt.hour
    month = df['ds'].dt.month

    # Daily solar curve (bell curve peaking at 13:00)
    daily = np.exp(-0.5 * ((hour - 13) / 3.5) ** 2)

    # Seasonal factor (higher in summer/post-monsoon)
    seasonal = 0.75 + 0.25 * np.sin((month - 3) * np.pi / 6)

    # Random cloud cover noise
    noise = np.random.normal(1.0, 0.15, len(df)).clip(0.2, 1.3)

    df['y'] = (daily * seasonal * noise * 850).clip(0)  # MW, max ~850MW
    df['y'] = df['y'].round(2)

    return df

if __name__ == "__main__":
    df = generate_solar_data()
    df.to_csv("data/solar_history.csv", index=False)
    print(f"Generated {len(df)} rows of solar data")
    print(df.tail())