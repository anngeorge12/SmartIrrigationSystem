import requests

def get_weather():

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": 13.0,
        "longitude": 77.6,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
        "timezone": "auto"
    }

    response = requests.get(url, params=params)
    data = response.json()

    weather = {
        "temp_c": data["daily"]["temperature_2m_max"][0],
        "temperature": data["daily"]["temperature_2m_max"][0],
        "rain_mm": data["daily"]["precipitation_sum"][0],
        "rainfall": data["daily"]["precipitation_sum"][0],
        "wind_speed": data["daily"]["windspeed_10m_max"][0],
        "humidity": 50   # placeholder until we add humidity API
    }

    return weather