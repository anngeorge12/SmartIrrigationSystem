import ee
from datetime import datetime, timedelta

# initialize earth engine
ee.Initialize(project='samrtirrigation-489614')


def get_satellite_data():

    # farm location (Bangalore example)
    point = ee.Geometry.Point([77.6, 13.0])

    # last 30 days range
    end_date = datetime.today()
    start_date = end_date - timedelta(days=30)

    dataset = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
        .first()
    )

    # NDVI
    ndvi = dataset.normalizedDifference(['B8', 'B4']).rename('NDVI')

    # NDWI
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
        "NDVI": list(ndvi_value.values())[0],
        "NDWI": list(ndwi_value.values())[0]
    }

    return result