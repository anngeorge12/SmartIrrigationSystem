import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("LSTM_TRAINING_DATASET.csv")

# Features used for LSTM
features = [
    "temp_c","swvl1","swvl2","solar_MJ","evap_mm","rain_mm","sp",
    "soil_avg","rain_3d","soil_lag1",
    "evapotranspiration","humidity","solar_radiation","temperature",
    "rainfall","wind_speed","rain_event","dryness_index",
    "heat_stress","month","NDVI","NDWI"
]

target = "irrigation_mm"

sequence_length = 7

X = []
y = []

for i in range(len(df) - sequence_length):

    seq = df[features].iloc[i:i+sequence_length].values
    label = df[target].iloc[i+sequence_length]

    X.append(seq)
    y.append(label)

X = np.array(X)
y = np.array(y)

print("Input shape:", X.shape)
print("Output shape:", y.shape)

# Save sequences
np.savez("lstm_sequences.npz", X=X, y=y)

print("LSTM sequences saved")