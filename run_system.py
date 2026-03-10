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