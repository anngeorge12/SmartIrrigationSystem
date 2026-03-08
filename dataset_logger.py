import csv
from datetime import datetime

def log_data(features):

    file = "irrigation_dataset.csv"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    columns = [
    "temp_c","swvl1","swvl2","solar_MJ","evap_mm","rain_mm","sp","soil_avg",
    "rain_3d","soil_lag1","evapotranspiration","humidity","solar_radiation",
    "temperature","rainfall","wind_speed","rain_event","dryness_index",
    "heat_stress","month","NDVI","NDWI"
    ]

    row = [timestamp] + [features[c] for c in columns]

    with open(file,"a",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print("Data saved to dataset.")