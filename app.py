from flask import Flask, request, jsonify
import os

from feature_engineering import build_features
from  predict_irrigation import predict_irrigation
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
# API endpoint
# -------------------------
@app.route("/predict", methods=["GET"])
def predict():

    try:
        # Get IoT data
        temp = float(request.args.get("temp"))
        hum = float(request.args.get("hum"))
        soil = float(request.args.get("soil"))

        iot_data = {
            "temp": temp,
            "hum": hum,
            "soil": soil
        }

        # Fetch external data
        weather = get_weather()
        satellite = get_satellite_data()

        # Build features
        features = build_features(weather, satellite, iot=iot_data)

        # Log data
        log_data(features)

        # RF decision
        rf_result = predict_irrigation(features)

        if rf_result["prediction"] == 0:
            return jsonify({
                "pump": 0,
                "duration": 0
            })

        # LSTM prediction
        irrigation_mm = predict_irrigation_amount(features)
        duration = mm_to_seconds(irrigation_mm)

        return jsonify({
            "pump": 1,
            "duration": duration,
            "mm": irrigation_mm
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    #app.run(host="0.0.0.0", port=5000)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))