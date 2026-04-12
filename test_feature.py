from feature_engineering import build_features
from predict_irrigation import predict_irrigation
from predict_irrigation_amt import predict_irrigation_amount  # update filename
import pandas as pd




def mm_to_seconds(mm):
    flow_rate = 8   # mm per minute (can tune later)
    seconds = (mm / flow_rate) * 60
    return int(seconds)

# Dummy data
weather = {
    "temp_c": 42,
    "humidity": 15,
    "rain_mm": 0,
    "wind_speed": 8
}

satellite = {
    "NDVI": 0.1,
    "NDWI": -0.5
}

iot_data = {
    "temp": 40,
    "hum": 10,
    "soil": 3800
}

# Build features
features = build_features(weather, satellite, iot=iot_data)


print("\n--- FINAL FEATURES PASSED TO RF ---\n")
df = pd.DataFrame([features])
print(df.T)

# RF decision
rf_result = predict_irrigation(features)

print("\nRF RESULT:", rf_result)

# 👉 ONLY if irrigation needed
if rf_result["prediction"] == 1:

    irrigation_mm = predict_irrigation_amount(features)
    duration = mm_to_seconds(irrigation_mm)

    print("\n💧 LSTM RESULT:")
    print("Irrigation Required:", irrigation_mm, "mm")
    print("Pump Duration:", duration, "seconds")

else:
    print("\n🚫 No irrigation needed")