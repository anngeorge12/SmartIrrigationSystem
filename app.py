from flask import Flask, request, jsonify

from feature_engineering import build_features
from predict_irrigation import predict_irrigation
from predict_irrigation_amt import predict_irrigation_amount

from weather_fetch import get_weather
from satellite_fetch import get_satellite_data
from dataset_logger import log_data

app = Flask(__name__)

# -------------------------
# Conversion function
# -------------------------
def mm_to_seconds(mm):
    flow_rate = 8
    return int((mm / flow_rate) * 60)

# -------------------------
# Health check route
# -------------------------
@app.route("/", methods=["GET"])
def home():
    return "Server running"

# -------------------------
# API endpoint
# -------------------------
@app.route("/predict", methods=["GET"])
def predict():

    try:
        # -------------------------
        # Get IoT data
        # -------------------------
        temp = float(request.args.get("temp"))
        hum = float(request.args.get("hum"))
        soil = float(request.args.get("soil"))

        print("\n📡 Incoming Data:", temp, hum, soil)

        iot_data = {
            "temp": temp,
            "hum": hum,
            "soil": soil
        }

        # -------------------------
        # Fetch external data
        # -------------------------
        weather = get_weather()
        satellite = get_satellite_data()

        # -------------------------
        # Build features
        # -------------------------
        features = build_features(weather, satellite, iot=iot_data)

        print("\n📊 FEATURES:")
        for k, v in features.items():
            print(f"{k}: {v}")

        # -------------------------
        # RF decision
        # -------------------------
        rf_result = predict_irrigation(features)

        print("\n🌲 RF RESULT:", rf_result)

        # -------------------------
        # RF threshold decision
        # -------------------------
        if rf_result["prediction"] == 0:
            print("🚫 RF DECISION: No irrigation needed")

            log_data(features)

            return jsonify({
                "pump": 0,
                "duration": 0,
                "source": "rf_block"
            })

        print("✅ RF DECISION: Irrigation needed → Calling LSTM")

        # -------------------------
        # LSTM prediction
        # -------------------------
        irrigation_mm = predict_irrigation_amount(features)

        # Safety clamp
        #irrigation_mm = max(0.5, min(irrigation_mm, 30))

        duration = mm_to_seconds(irrigation_mm)

        #print("💧 LSTM OUTPUT (mm):", irrigation_mm)

        # Log AFTER prediction
        log_data(features)

        return jsonify({
            "pump": 1,
            "duration": duration,
            "mm": irrigation_mm,
            "source": "lstm"
        })

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({
            "error": str(e)
        })


# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)