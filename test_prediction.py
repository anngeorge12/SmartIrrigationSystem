from predict_irrigation import predict_irrigation

features = {
    "temp_c":33,
    "swvl1":0.28,
    "swvl2":0.38,
    "solar_MJ":2.13,
    "evap_mm":1.65,
    "rain_mm":0,
    "sp":101325,
    "soil_avg":0.33,
    "rain_3d":0,
    "soil_lag1":0.33,
    "evapotranspiration":1.65,
    "humidity":50,
    "solar_radiation":24.75,
    "temperature":33,
    "rainfall":0,
    "wind_speed":11,
    "rain_event":0,
    "dryness_index":28,
    "heat_stress":0,
    "month":3,
    "NDVI":0.27,
    "NDWI":-0.04
}

result = predict_irrigation(features)

print(result)