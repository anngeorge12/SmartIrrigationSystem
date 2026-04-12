import joblib
import pandas as pd

# ======================
# LOAD MODEL + SCALER
# ======================

model = joblib.load("irrigation_model_aftershap.pkl")
scaler = joblib.load("scaler_aftershap.pkl")

# ======================
# FEATURE ORDER
# MUST MATCH TRAINING
# ======================

FEATURE_COLUMNS = [
    "temp_c",
    "swvl1",
    "swvl2",
    "solar_MJ",
    "evap_mm",
    "rain_mm",
    "sp",
    "soil_avg",
    "rain_3d",
    "soil_lag1",
    "evapotranspiration",
    "humidity",
    "solar_radiation",
    "temperature",
    "rainfall",
    "wind_speed",
    "rain_event",
    "dryness_index",
    "heat_stress",
    "month",
    "NDVI",
    "NDWI"
]


# ======================
# PREDICTION FUNCTION
# ======================

def predict_irrigation(features_dict):

    # Convert to dataframe
    df = pd.DataFrame([features_dict])

    # Ensure correct feature order
    df = df[FEATURE_COLUMNS]

    # Apply scaler
    X_scaled = scaler.transform(df)

    # Model prediction
    prediction = model.predict(X_scaled)[0]

    # Probability (useful for dashboard)
    probability = model.predict_proba(X_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "irrigation_probability": float(probability)
    }