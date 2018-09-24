import fnmatch
import json
import geopandas
import rasterio
from rasterio import mask
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import os
from os import listdir
from os.path import isfile, join
import sys

ROOT_DIR = os.path.join(os.getcwd(), "..")
SENTINELPRODUCTS_DIR = os.path.join(ROOT_DIR, "sentineldata", "products")
GEOJSON_DIR = os.path.join(ROOT_DIR, "datasets", "crop-disease")
SENTINELDATA_DIR = os.path.join(SENTINELPRODUCTS_DIR, "data")
REGION_DATA_FILE = os.path.join(GEOJSON_DIR, "regions.json")


url = "https://scihub.copernicus.eu/dhus"
user = sys.argv[1]
pw = sys.argv[2]

api = SentinelAPI(user, pw, url)

def download_sentinel_products_for_ROI(geojson_file):
    print("Searching products for %s" % geojson_file)
    geojson = read_geojson(geojson_file)
    if geojson.features[0].properties.get("id") is None:
        # generate id
        import uuid
        geojson.features[0].properties["id"] = str(uuid.uuid4())

    footprint = geojson_to_wkt(geojson)

    feature = geojson.features[0]
    date = feature.properties["date"]
    incubation = feature.properties["incubation"]
    
    # TODO adjustable coverage interval
    min_cloudcoverpercentage = 0
    max_cloudcoverpercentage = 100
    cloudcoverpercentage = min_cloudcoverpercentage

    while True:
        products = api.query(footprint,
                             date=(date + "-" + incubation, date),
                             platformname="Sentinel-2",
                             filename="S2A_*",
                             cloudcoverpercentage=(min_cloudcoverpercentage, cloudcoverpercentage)
                             )
        if len(products) > 0:
            print("Found %d Products, downloading %d GB" % (len(products), api.get_products_size(products)))
            break
        elif len(products) == 0:
            # if no products found, end programm
            if cloudcoverpercentage >= max_cloudcoverpercentage:
                print("Found no products for specified search terms.")
                exit(1)
            cloudcoverpercentage += 5
            print("Found no products. Increasing cloud coverage to %d %%" % cloudcoverpercentage)

    if not os.path.exists(SENTINELPRODUCTS_DIR):
        os.makedirs(SENTINELPRODUCTS_DIR)

    try:
        api.download_all(list(products.keys()), SENTINELPRODUCTS_DIR)
    except ConnectionError as err:
        print(err)

    # map geojson to product IDs for the training
    geojson.features[0].properties["sentinelproducts"] = list(map(lambda item: item["filename"], products.values()))

    if not os.path.exists(REGION_DATA_FILE):
        open(REGION_DATA_FILE, "w").close()

    # collect all features in a single file
    with open(REGION_DATA_FILE, "r") as f:
        try:
            regions = json.load(f)
            regions["features"].append(geojson.features[0])
        except ValueError:
            regions = geojson

    with open(REGION_DATA_FILE, "w") as f:
        json.dump(regions, f)

    # unzip all products
    files = get_files_in_path(SENTINELPRODUCTS_DIR)
    if not os.path.exists(SENTINELDATA_DIR):
        os.mkdir(SENTINELDATA_DIR)

    for file in files:
        import zipfile

        with zipfile.ZipFile(os.path.join(SENTINELPRODUCTS_DIR, file)) as zip_ref:
            try:
                zip_ref.extractall(path=SENTINELDATA_DIR)
            except Exception:
                print(zip_ref + " already extracted. Skip file.")
                continue

    return api.to_geodataframe(products)


def get_files_in_path(path):
    return [f for f in listdir(path) if isfile(join(path, f))]


geoms = []
files = get_files_in_path(GEOJSON_DIR)

for file in files:
    if file.endswith(".geojson"):
        geojson_file = os.path.join(GEOJSON_DIR, file)
        download_sentinel_products_for_ROI(geojson_file)


# calculate ndvi
def getndvi(nir_band, red_band):
    return (nir_band.astype(rasterio.float32) - red_band.astype(rasterio.float32)) / (nir_band + red_band)


def create_bb_data_frame(left, bottom, right, top):
    """
    returns a geopandas.GeoDataFrame bounding box
    """
    from shapely.geometry import Point, Polygon

    p1 = Point(left, top)
    p2 = Point(right, top)
    p3 = Point(right, bottom)
    p4 = Point(left, bottom)

    np1 = (p1.coords.xy[0][0], p1.coords.xy[1][0])
    np2 = (p2.coords.xy[0][0], p2.coords.xy[1][0])
    np3 = (p3.coords.xy[0][0], p3.coords.xy[1][0])
    np4 = (p4.coords.xy[0][0], p4.coords.xy[1][0])

    bb_polygon = Polygon([np1, np2, np3, np4])

    return geopandas.GeoDataFrame(geopandas.GeoSeries(bb_polygon), columns=['geometry'])


NDVI_DIR = os.path.join(SENTINELPRODUCTS_DIR, "ndvi")
if not os.path.exists(NDVI_DIR):
    os.mkdir(NDVI_DIR)

geojson = read_geojson(REGION_DATA_FILE)
for feature in geojson.features:
    from geojson import FeatureCollection
    import fiona
    geom = geopandas.GeoDataFrame.from_features(FeatureCollection([feature]))

    if not geom.crs:
        geom.crs = fiona.crs.from_epsg(4326)

    for product in feature.properties["sentinelproducts"]:
        for root, dir_names, file_names in os.walk(os.path.join(SENTINELDATA_DIR, product)):
            sorted_files = sorted(list(filter(lambda item:
                                                    # L1C products or
                                                    item.endswith("B04.jp2") or item.endswith("B08.jp2")
                                                    # L2A products
                                                    or item.endswith("B04_10m.jp2") or item.endswith("B08_10m.jp2"),
                                              fnmatch.filter(file_names, '*.jp2'))))
            if len(sorted_files) == 0:
                continue;

            b4, b8 = list(map(lambda item: os.path.join(root, item), sorted_files))

            with rasterio.open(b4) as red:
                bounding_box = create_bb_data_frame(red.bounds.left, red.bounds.bottom, red.bounds.right, red.bounds.top)
                # polygons from geojson to target crs
                mapped_geom = geom.to_crs(red.crs)
                # ensure that all polygons intersect bounding box
                filtered_polygons = list(filter(lambda item: bounding_box.intersects(item).bool(),
                                                mapped_geom.geometry))
                polygons = list(map(lambda item: json.loads(geopandas.GeoSeries(item).to_json())["features"][0]["geometry"],
                                    filtered_polygons))
                # get mask from geojson data
                red_cropped_mask, red_cropped_transform = mask.mask(red, polygons, crop=True)
                profile = red.meta.copy()

            with rasterio.open(b8) as nir:
                nir_cropped_mask, nir_cropped_transform = mask.mask(nir, polygons, crop=True)

            ndvi = getndvi(nir_cropped_mask, red_cropped_mask)

            profile.update({"driver":    "GTiff",
                            "dtype":     rasterio.float32,
                            "height":    red_cropped_mask.shape[1],
                            "width":     red_cropped_mask.shape[2],
                            "transform": red_cropped_transform})

            ndvi_file = os.path.join(NDVI_DIR, feature.properties["id"] + "_" + product + ".tif")
            ndvi_mask_file = os.path.join(NDVI_DIR, feature.properties["id"] + "_" + product + ".mask")
            with rasterio.open(ndvi_file, "w", **profile) as dst:
                dst.write(ndvi)
                
            # save ndvi values for later use
            ndvi.dump(ndvi_mask_file)
