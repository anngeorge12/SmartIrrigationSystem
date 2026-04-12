import ee
import os
import json
import tempfile
from datetime import datetime, timedelta


def initialize_ee():
    try:
        service_account = os.environ.get("EE_SERVICE_ACCOUNT")
        key_json = os.environ.get("EE_PRIVATE_KEY")

        # 🔐 CLOUD MODE (Render)
        if service_account and key_json:
            key_dict = json.loads(key_json)

            # write JSON to temporary file
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
                json.dump(key_dict, f)
                key_path = f.name

            credentials = ee.ServiceAccountCredentials(
                service_account, key_path
            )
            ee.Initialize(credentials)

            print("✅ Earth Engine initialized (Service Account)")

        # 💻 LOCAL MODE (VS Code)
        else:
            ee.Initialize(project='samrtirrigation-489614')
            print("⚠️ Using local Earth Engine authentication")

    except Exception as e:
        print("❌ EE Initialization failed:", e)


# initialize once
initialize_ee()


def get_satellite_data():
    try:
        # 📍 Farm location (Bangalore example)
        point = ee.Geometry.Point([77.6, 13.0])

        # 📅 Last 30 days
        end_date = datetime.today()
        start_date = end_date - timedelta(days=30)

        dataset = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(point)
            .filterDate(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            .sort("CLOUDY_PIXEL_PERCENTAGE")
            .first()
        )

        # 🚨 Safety check
        if dataset is None:
            print("⚠️ No satellite data found → using fallback")
            return {"NDVI": 0.2, "NDWI": -0.1}

        # 🌿 NDVI
        ndvi = dataset.normalizedDifference(['B8', 'B4']).rename('NDVI')

        # 💧 NDWI
        ndwi = dataset.normalizedDifference(['B8', 'B11']).rename('NDWI')

        ndvi_value = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=10
        ).getInfo()

        ndwi_value = ndwi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=10
        ).getInfo()

        result = {
            "NDVI": list(ndvi_value.values())[0] if ndvi_value else 0.2,
            "NDWI": list(ndwi_value.values())[0] if ndwi_value else -0.1
        }

        # 🔍 Debug print
        print("NDVI:", result["NDVI"])
        print("NDWI:", result["NDWI"])

        return result

    except Exception as e:
        print("❌ Satellite fetch failed:", e)
        return {"NDVI": 0.2, "NDWI": -0.1}