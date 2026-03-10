import pandas as pd

# Load historical dataset
df = pd.read_csv("FINAL_DEPLOYMENT_DATASET.csv")

# Function to generate irrigation amount
def calculate_irrigation(row):
    
    if row["irrigate"] == 1:
        return max(row["dryness_index"] * 0.35, 1)
    else:
        return 0

# Create new column
df["irrigation_mm"] = df.apply(calculate_irrigation, axis=1)

# KEEP ONLY IRRIGATION DAYS
df = df[df["irrigation_mm"] > 0]

# Save new dataset
df.to_csv("LSTM_TRAINING_DATASET.csv", index=False)

print("LSTM dataset created successfully")
print(df[["dryness_index","irrigate","irrigation_mm"]].head())
print("Rows remaining:", len(df))