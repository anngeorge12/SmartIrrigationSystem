import joblib

model = joblib.load("irrigation_model_aftershap.pkl")
scaler = joblib.load("scaler_aftershap.pkl")

print("Model loaded successfully")
print(type(model))