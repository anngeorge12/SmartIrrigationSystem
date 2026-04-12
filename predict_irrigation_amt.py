import pandas as pd
import numpy as np
import os
import logging

# ✅ Setup logging (important for Render)
logging.basicConfig(level=logging.INFO)

# --- TENSORFLOW MODEL LOADING ---
try:
    from tensorflow.keras.models import load_model

    model = load_model("model_fixed.keras")
    MODEL_AVAILABLE = True

    logging.info("✅ LSTM model loaded successfully")

except Exception as e:
    MODEL_AVAILABLE = False
    logging.error(f"❌ TensorFlow Load FAILED: {e}")
    logging.error("🚨 SYSTEM USING FALLBACK — LSTM NOT ACTIVE")


def predict_irrigation_amount(features):

    # --- IF LSTM IS AVAILABLE ---
    if MODEL_AVAILABLE:
        try:
            logging.info("🤖 Using LSTM model for prediction")

            # Load datasets
            hist = pd.read_csv("LSTM_TRAINING_DATASET.csv")
            live = pd.read_csv("irrigation_dataset.csv")

            # Process timestamps
            live["timestamp"] = pd.to_datetime(live["timestamp"])
            live["date"] = live["timestamp"].dt.date
            live_grouped = live.groupby("date").mean().reset_index()

            # Feature columns
            feature_cols = [
                "temp_c","swvl1","swvl2","solar_MJ","evap_mm","rain_mm","sp",
                "soil_avg","rain_3d","soil_lag1",
                "evapotranspiration","humidity","solar_radiation","temperature",
                "rainfall","wind_speed","rain_event","dryness_index",
                "heat_stress","month","NDVI","NDWI"
            ]

            # Build 7-day sequence
            if len(live_grouped) >= 7:
                seq_df = live_grouped[feature_cols].tail(7)
            else:
                needed = 7 - len(live_grouped)
                hist_part = hist[feature_cols].tail(needed)
                live_part = live_grouped[feature_cols]
                seq_df = pd.concat([hist_part, live_part])

            # Prepare input
            X = np.expand_dims(seq_df.values, axis=0)

            # Predict
            irrigation = model.predict(X, verbose=0)[0][0]

            logging.info(f"💧 LSTM Prediction (mm): {irrigation}")

            return float(irrigation)

        except Exception as e:
            logging.error(f"❌ LSTM prediction error: {e}")
            logging.warning("⚠️ Falling back to formula")

    # --- FALLBACK LOGIC ---
    logging.warning("⚠️ Using fallback formula instead of LSTM")

    soil_val = features.get('soil_avg', 0.15)
    dryness = features.get('dryness_index', 30)

    calc_amount = (max(0, 0.35 - soil_val) * 25) + (dryness / 15)

    result = float(round(max(0.8, calc_amount), 2))

    logging.info(f"💧 Fallback Prediction (mm): {result}")

    return result