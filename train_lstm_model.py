import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==============================
# 1️⃣ Load prepared sequences
# ==============================

data = np.load("lstm_sequences.npz")

X = data["X"]
y = data["y"]

print("Dataset shape:", X.shape, y.shape)

# ==============================
# 2️⃣ Time-aware train/test split
# ==============================

split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ==============================
# 3️⃣ Build LSTM model
# ==============================

model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(32))
model.add(Dense(16, activation="relu"))

model.add(Dense(1))

# ==============================
# 4️⃣ Compile model
# ==============================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# ==============================
# 5️⃣ Train model
# ==============================

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# ==============================
# 6️⃣ Evaluate model
# ==============================

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)

print("\nLSTM Performance")
print("-------------------")
print("MAE:", mae)
print("RMSE:", rmse)

# ==============================
# 7️⃣ Plot predictions
# ==============================

plt.figure(figsize=(10,5))

plt.plot(y_test[:50], label="Actual")
plt.plot(preds[:50], label="Predicted")

plt.title("Irrigation Prediction (First 50 Samples)")
plt.xlabel("Samples")
plt.ylabel("Irrigation mm")

plt.legend()
plt.show()

# ==============================
# 8️⃣ Save model
# ==============================

model.save("irrigation_lstm_model.h5")

print("\nLSTM model saved successfully")