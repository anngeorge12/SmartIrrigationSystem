import pandas as pd
import numpy as np
import logging

# =========================
# SETUP LOGGING
# =========================
logging.basicConfig(level=logging.INFO)

# =========================
# LOAD LSTM MODEL
# =========================
try:
    from tensorflow.keras.models import load_model

    model = load_model("irrigation_lstm_model.h5", compile=False)
    MODEL_AVAILABLE = True

    logging.info("✅ LSTM model loaded successfully")

except Exception as e:
    MODEL_AVAILABLE = False
    logging.error(f"❌ TensorFlow Load FAILED: {e}")
    logging.error("🚨 SYSTEM USING FALLBACK — LSTM NOT ACTIVE")


# =========================
# FEATURE COLUMNS
# =========================
FEATURE_COLS = [
    "temp_c","swvl1","swvl2","solar_MJ","evap_mm","rain_mm","sp",
    "soil_avg","rain_3d","soil_lag1",
    "evapotranspiration","humidity","solar_radiation","temperature",
    "rainfall","wind_speed","rain_event","dryness_index",
    "heat_stress","month","NDVI","NDWI"
]

# =========================
# LOAD TRAINING DATA STATS
# =========================
train_df = pd.read_csv("LSTM_TRAINING_DATASET.csv")

feature_min = train_df[FEATURE_COLS].min()
feature_max = train_df[FEATURE_COLS].max()


# =========================
# MAIN FUNCTION
# =========================
def predict_irrigation_amount(features):

    if MODEL_AVAILABLE:
        try:
            logging.info("🤖 Using LSTM model for prediction")

            # -------------------------
            # LOAD DATA
            # -------------------------
            hist = pd.read_csv("LSTM_TRAINING_DATASET.csv")
            live = pd.read_csv("irrigation_dataset.csv")

            live["timestamp"] = pd.to_datetime(live["timestamp"])
            live["date"] = live["timestamp"].dt.date

            live_grouped = live.groupby("date").mean().reset_index()

            # -------------------------
            # ADD CURRENT REAL-TIME DATA
            # -------------------------
            current_df = pd.DataFrame([features])[FEATURE_COLS]
            current_df["date"] = pd.to_datetime("today").date()

            live_grouped = pd.concat([live_grouped, current_df], ignore_index=True)

            # -------------------------
            # BUILD SEQUENCE (BOOST CURRENT IMPACT)
            # -------------------------
            seq_df = live_grouped[FEATURE_COLS].tail(6)

            # Add current row twice (important)
            seq_df = pd.concat([seq_df, current_df[FEATURE_COLS]])
            seq_df = pd.concat([seq_df, current_df[FEATURE_COLS]])

            # -------------------------
            # DEBUG INPUT
            # -------------------------
            print("\n🧪 FULL LSTM INPUT SEQUENCE:")
            print(seq_df)

            # -------------------------
            # NORMALIZATION (CRITICAL FIX)
            # -------------------------
            seq_df_norm = (seq_df - feature_min) / (feature_max - feature_min)
            seq_df_norm = seq_df_norm.fillna(0)

            # -------------------------
            # PREPARE INPUT
            # -------------------------
            X = np.expand_dims(seq_df_norm.values, axis=0)

            # -------------------------
            # PREDICT
            # -------------------------
            irrigation = model.predict(X, verbose=0)[0][0]

            logging.info(f"💧 LSTM Prediction (mm): {irrigation}")

            return float(irrigation)

        except Exception as e:
            logging.error(f"❌ LSTM prediction error: {e}")
            logging.warning("⚠️ Falling back to formula")

    # =========================
    # FALLBACK
    # =========================
    logging.warning("⚠️ Using fallback formula instead of LSTM")

    soil_val = features.get('soil_avg', 0.15)
    dryness = features.get('dryness_index', 30)

    calc_amount = (max(0, 0.35 - soil_val) * 25) + (dryness / 15)

    result = float(round(max(0.8, calc_amount), 2))

    logging.info(f"💧 Fallback Prediction (mm): {result}")

    return result