import ee

ee.Initialize()

def get_satellite_data():

    point = ee.Geometry.Point([77.6, 13.0])

    dataset = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(point)
        .filterDate("2024-01-01", "2024-12-31")
        .sort("CLOUDY_PIXEL_PERCENTAGE")
        .first()
    )

    ndvi = dataset.normalizedDifference(['B8','B4']).rename('NDVI')
    ndwi = dataset.normalizedDifference(['B8','B11']).rename('NDWI')

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

    return {
        "NDVI": list(ndvi_value.values())[0],
        "NDWI": list(ndwi_value.values())[0]
    }