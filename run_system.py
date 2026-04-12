DEMO_MODE = False

from weather_fetch import get_weather
from satellite_fetch import get_satellite_data
from feature_engineering import build_features
from dataset_logger import log_data
from predict_irrigation import predict_irrigation
from predict_irrigation_amt import predict_irrigation_amount
import pandas as pd

print("Fetching weather data...")
weather = get_weather()

print("Fetching satellite data...")
satellite = get_satellite_data()

print("Building features...")
features = build_features(weather, satellite)

# ======================
# DEMO MODE SIMULATION
# ======================
if DEMO_MODE:
    print("\nDemo mode enabled: simulating drought conditions")
    features.update({
        "dryness_index": 60,
        "humidity": 15,
        "swvl1": 0.05,
        "swvl2": 0.08,
        "soil_avg": 0.065,
        "soil_lag1": 0.065,
        "NDVI": 0.10,
        "NDWI": -0.20,
        "rain_mm": 0,
        "rainfall": 0,
        "rain_3d": 0
    })

# ==============================
# AI PREDICTION LOGIC
# ==============================
result = predict_irrigation(features)
rf_prediction = result["prediction"]

# 1. Add the prediction to the features dictionary so it gets logged
features["irrigation_prediction"] = rf_prediction

print("\nAI Irrigation Decision")
print("----------------------")

if rf_prediction == 0:
    print("Irrigation Needed: NO")
    print("Recommended Water: 0 mm")
    features["lstm_water_mm"] = 0.0 # Add to features for logging
else:
    irrigation_mm = predict_irrigation_amount(features)
    print("Irrigation Needed: YES")
    print("Recommended Water:", round(irrigation_mm, 2), "mm")
    features["lstm_water_mm"] = round(float(irrigation_mm), 2) # Add to features for logging

# ==============================
# DATA LOGGING & EXPORT
# ==============================
# Now log_data will save the features PLUS the new AI columns
log_data(features) 

# Update the JSON for the dashboard
df = pd.read_csv("irrigation_dataset.csv")
df.to_json("irrigation_dataset.json", orient="records")

print("\nSystem execution complete. Dashboard data updated.")