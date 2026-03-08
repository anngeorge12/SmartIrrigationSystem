from weather_fetch import get_weather
from satellite_fetch import get_satellite_data
from feature_engineering import build_features

print("Fetching weather data...")
weather = get_weather()

print("Fetching satellite data...")
satellite = get_satellite_data()

print("Building features...")
features = build_features(weather, satellite)

print("Current features:")
for k,v in features.items():
    print(k,":",v)

print("Model prediction step will be added later.")