from weather_fetch import get_weather
from satellite_fetch import get_satellite_data
from feature_engineering import build_features
from dataset_logger import log_data

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

print("Model prediction step will be added later.")