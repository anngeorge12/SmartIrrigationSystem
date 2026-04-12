import ee
import os
import json
import tempfile
import logging
from datetime import datetime, timedelta

# ✅ setup logging (important for Render)
logging.basicConfig(level=logging.INFO)


def initialize_ee():
    try:
        service_account = os.environ.get("EE_SERVICE_ACCOUNT")
        key_json = os.environ.get("EE_PRIVATE_KEY")

        if service_account and key_json:
            key_dict = json.loads(key_json)

            # write JSON to temp file
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
                json.dump(key_dict, f)
                key_path = f.name

            credentials = ee.ServiceAccountCredentials(
                service_account, key_path
            )
            ee.Initialize(credentials)

            logging.info("✅ Earth Engine initialized (Service Account)")

        else:
            ee.Initialize(project='samrtirrigation-489614')
            logging.warning("⚠️ Using local Earth Engine authentication")

    except Exception as e:
        logging.error(f"❌ EE Initialization failed: {e}")


# initialize once
initialize_ee()


def get_satellite_data():
    try:
        logging.info("🚀 Satellite function started")

        # 📍 Location (Bangalore)
        point = ee.Geometry.Point([77.6, 13.0])

        # 📅 Date range (last 30 days)
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

        # 🚨 Check dataset
        if dataset is None:
            logging.warning("⚠️ No dataset found → fallback used")
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

        # 🚨 Validate values
        if not ndvi_value or not ndwi_value:
            logging.warning("⚠️ Empty satellite values → fallback used")
            return {"NDVI": 0.2, "NDWI": -0.1}

        ndvi_val = list(ndvi_value.values())[0]
        ndwi_val = list(ndwi_value.values())[0]

        logging.info(f"🌿 NDVI VALUE: {ndvi_val}")
        logging.info(f"💧 NDWI VALUE: {ndwi_val}")

        return {
            "NDVI": ndvi_val,
            "NDWI": ndwi_val
        }

    except Exception as e:
        logging.error(f"❌ Satellite fetch failed: {e}")
        return {"NDVI": 0.2, "NDWI": -0.1}