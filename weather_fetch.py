import requests
import pandas as pd
import os

print("Fetching weather data...")

latitude = 12.97
longitude = 77.59

url = "https://api.open-meteo.com/v1/forecast"

params = {
    "latitude": latitude,
    "longitude": longitude,
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "windspeed_10m_max"
    ],
    "timezone": "auto"
}

response = requests.get(url, params=params)
data = response.json()

daily = data["daily"]

today_data = {
    "date": daily["time"][0],
    "temp_max": daily["temperature_2m_max"][0],
    "temp_min": daily["temperature_2m_min"][0],
    "rainfall": daily["precipitation_sum"][0],
    "wind_speed": daily["windspeed_10m_max"][0]
}

df_today = pd.DataFrame([today_data])

print("\nToday's Weather Data")
print(df_today)

file_name = "weather_dataset.csv"

# Check if dataset already exists
if os.path.exists(file_name):
    
    df_existing = pd.read_csv(file_name)

    # Avoid duplicate dates
    if today_data["date"] not in df_existing["date"].values:
        df_updated = pd.concat([df_existing, df_today], ignore_index=True)
        df_updated.to_csv(file_name, index=False)
        print("\nWeather dataset updated.")
    else:
        print("\nToday's data already exists in dataset.")

else:
    
    df_today.to_csv(file_name, index=False)
    print("\nNew weather dataset created.")




    