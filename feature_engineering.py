import pandas as pd
from datetime import datetime
import os

def build_features(weather, satellite, iot=None):

    temp = weather["temp_c"]
    humidity = weather["humidity"]
    rainfall = weather["rain_mm"]

# Override with IoT if available
    if iot:
        temp = iot.get("temp", temp)
        humidity = iot.get("hum", humidity)

    features = {}

    # basic weather features
    features["temp_c"] = temp
    features["temperature"] = temp
    features["rain_mm"] = rainfall
    features["rainfall"] = rainfall
    features["wind_speed"] = weather["wind_speed"]
    features["humidity"] = humidity

    # satellite features
    features["NDVI"] = satellite["NDVI"]
    features["NDWI"] = satellite["NDWI"]

    # derived features
    features["month"] = datetime.now().month
    features["rain_event"] = 1 if rainfall > 0 else 0
    features["heat_stress"] = 1 if temp > 35 else 0

    # soil moisture conversion
    if iot and "soil" in iot:
        soil_moisture = 1- iot["soil"] / 4095   # normalize ESP32 value
        soil_moisture = max(0.05, min(0.6, soil_moisture))
    else:
        ndwi = satellite["NDWI"]
        soil_moisture = (ndwi + 1) / 2

    features["swvl1"] = soil_moisture * 0.6
    features["swvl2"] = soil_moisture * 0.8
    features["soil_avg"] = (features["swvl1"] + features["swvl2"]) / 2

    # other estimated variables
    features["solar_radiation"] = temp * 0.75
    features["solar_MJ"] = features["solar_radiation"] * 0.0864
    features["evapotranspiration"] = temp * 0.05
    features["evap_mm"] = features["evapotranspiration"]
    features["sp"] = 101325
    features["dryness_index"] = temp - weather["humidity"]/10

    # --- read dataset history ---
    dataset_file = "irrigation_dataset.csv"

    if os.path.exists(dataset_file):

        df = pd.read_csv(dataset_file)

        if len(df) == 0:
            features["rain_3d"] = rainfall
            features["soil_lag1"] = features["soil_avg"]

        else:

            # rain last 3 entries
            last_rain = df["rain_mm"].tail(2).sum()
            features["rain_3d"] = rainfall + last_rain

            # yesterday soil moisture
            features["soil_lag1"] = df["soil_avg"].iloc[-1]

    else:

        features["rain_3d"] = rainfall
        features["soil_lag1"] = features["soil_avg"]

    return features