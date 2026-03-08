from datetime import datetime

def build_features(weather, satellite):

    temp = weather["temp_c"]
    rainfall = weather["rain_mm"]

    features = {}

    features["temp_c"] = temp
    features["temperature"] = temp
    features["rain_mm"] = rainfall
    features["rainfall"] = rainfall
    features["wind_speed"] = weather["wind_speed"]
    features["humidity"] = weather["humidity"]

    features["NDVI"] = satellite["NDVI"]
    features["NDWI"] = satellite["NDWI"]

    # derived features
    features["month"] = datetime.now().month
    features["rain_event"] = 1 if rainfall > 0 else 0
    features["heat_stress"] = 1 if temp > 35 else 0

    features["swvl1"] = satellite["NDWI"] * 0.6
    features["swvl2"] = satellite["NDWI"] * 0.8

    features["soil_avg"] = (features["swvl1"] + features["swvl2"]) / 2
    features["soil_lag1"] = features["soil_avg"]

    features["solar_radiation"] = temp * 0.75
    features["solar_MJ"] = features["solar_radiation"] * 0.0864

    features["evapotranspiration"] = temp * 0.05
    features["evap_mm"] = features["evapotranspiration"]

    features["sp"] = 101325
    features["rain_3d"] = rainfall
    features["dryness_index"] = temp - weather["humidity"]/10

    return features