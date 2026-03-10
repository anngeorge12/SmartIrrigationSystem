import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("irrigation_lstm_model.h5", compile=False)

def predict_irrigation_amount(features):

    hist = pd.read_csv("LSTM_TRAINING_DATASET.csv")
    live = pd.read_csv("irrigation_dataset.csv")

    live["timestamp"] = pd.to_datetime(live["timestamp"])
    live["date"] = live["timestamp"].dt.date

    live = live.groupby("date").mean().reset_index()

    feature_cols = [
        "temp_c","swvl1","swvl2","solar_MJ","evap_mm","rain_mm","sp",
        "soil_avg","rain_3d","soil_lag1",
        "evapotranspiration","humidity","solar_radiation","temperature",
        "rainfall","wind_speed","rain_event","dryness_index",
        "heat_stress","month","NDVI","NDWI"
    ]

    # CASE 1: enough live data
    if len(live) >= 7:

        seq = live[feature_cols].tail(7).values
        print("Using last 7 days of live data")

    # CASE 2: not enough live data
    else:

        needed = 7 - len(live)

        hist_part = hist[feature_cols].tail(needed)
        live_part = live[feature_cols]

        seq = pd.concat([hist_part, live_part]).values

        print("Using historical + live data for LSTM")

    X = np.expand_dims(seq, axis=0)

    irrigation = model.predict(X)[0][0]

    return float(irrigation)