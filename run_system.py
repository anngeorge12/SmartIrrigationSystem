
DEMO_MODE = True  # Set to True to simulate drought conditions for testing

from weather_fetch import get_weather
from satellite_fetch import get_satellite_data
from feature_engineering import build_features
from dataset_logger import log_data
from predict_irrigation import predict_irrigation

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

    features["dryness_index"] = 60
    features["humidity"] = 15

    # simulate dry soil
    features["swvl1"] = 0.05
    features["swvl2"] = 0.08
    features["soil_avg"] = 0.065
    features["soil_lag1"] = 0.065

    # simulate stressed vegetation
    features["NDVI"] = 0.10
    features["NDWI"] = -0.20

    # no rainfall
    features["rain_mm"] = 0
    features["rainfall"] = 0
    features["rain_3d"] = 0

print("Current features:")
for k,v in features.items():
    print(k,":",v)

log_data(features)
result = predict_irrigation(features)

#print("\nAI Irrigation Decision")
#print("----------------------")
#print("Prediction:", result["prediction"])
#print("Probability:", result["irrigation_probability"])
#print("Model prediction step will be added later.")

from predict_irrigation_amt import predict_irrigation_amount

result = predict_irrigation(features)

print("\nAI Irrigation Decision")
print("----------------------")

rf_prediction = result["prediction"]

if rf_prediction == 0:

    print("Irrigation Needed: NO")
    print("Recommended Water: 0 mm")

else:

    irrigation_mm = predict_irrigation_amount(features)

    print("Irrigation Needed: YES")
    print("Recommended Water:", round(irrigation_mm,2), "mm")