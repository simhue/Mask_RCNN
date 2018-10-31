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
from geojson import FeatureCollection
import shutil

ROOT_DIR = os.path.join(os.getcwd(), "..")
SENTINELPRODUCTS_DIR = os.path.join(ROOT_DIR, "sentineldata", "products")
GEOJSON_DIR = os.path.join(ROOT_DIR, "datasets")
REGION_DATA_FILE = os.path.join(GEOJSON_DIR, "regions.geojson")
NDVI_DIR = os.path.join(SENTINELPRODUCTS_DIR, "ndvi")
TRAIN_DIR = os.path.join(GEOJSON_DIR, "train")
VAL_DIR = os.path.join(GEOJSON_DIR, "val") 

url = "https://scihub.copernicus.eu/dhus"
user = sys.argv[1]
pw = sys.argv[2]

api = SentinelAPI(user, pw, url)

def download_sentinel_products_for_ROI(geojson_file):
    print("Searching products for %s" % geojson_file)
    feature = read_geojson(geojson_file)

    footprint = geojson_to_wkt(feature.geometry)

    date = feature.properties["date"]
    incubation = feature.properties["incubation"]
    
    # TODO adjustable coverage interval
    # Config file?
    min_cloudcoverpercentage = 0
    max_cloudcoverpercentage = 100
    cloudcoverpercentage = max_cloudcoverpercentage

    products = api.query(footprint,
                         date=(date + "-" + incubation, date),
                         platformname="Sentinel-2",
                         cloudcoverpercentage=(min_cloudcoverpercentage, cloudcoverpercentage)
                         )
    if len(products) > 0:
        print("Found {} Products, downloading {} GB".format(len(products), api.get_products_size(products)))
    elif len(products) == 0:
        # if no products found, search for all avaivable products regardsless of cloudcoverage
        products = api.query(footprint,
                         date=(date + "-" + incubation, date),
                         platformname="Sentinel-2",
                         filename="S2A_*",
                         area_relation="intersects")
        
        if len(products) == 0:
            print("Found no products for specified search terms.")
        else:
            sorted_products = sorted([product["cloudcoverpercentage"] for product in products.values()], reverse=True)
            print("Found {} products with max. cloud coverage of {} %%".format(len(products), sorted_products[0]))


    if not os.path.exists(SENTINELPRODUCTS_DIR):
        os.makedirs(SENTINELPRODUCTS_DIR)

    try:
        api.download_all(list(products.keys()), SENTINELPRODUCTS_DIR)
    except ConnectionError as err:
        print(err)

    # for every found product, create a separate GeoJson Feature for later simplicity
    import uuid
    import copy
    regions = {}
    
    if not os.path.exists(REGION_DATA_FILE):
        regions = FeatureCollection([])
    else:
        with open(REGION_DATA_FILE, "r") as f:
            regions = json.load(f)
    
    for product in products.values():
        new_feature = copy.deepcopy(feature)
        new_feature.properties["id"] = str(uuid.uuid4())
        new_feature.properties["sentinelproduct"] = product["filename"]
        regions["features"].append(new_feature)
            
    with open(REGION_DATA_FILE, "w") as f:
        json.dump(regions, f)

def unzip_sentinel_products():
    files = get_files_in_path(SENTINELPRODUCTS_DIR)

    import zipfile
    import re
    
    for file in files:
        zip_ref = zipfile.ZipFile(os.path.join(SENTINELPRODUCTS_DIR, file))
        # only extract B04 and B08 bands
        for info in zip_ref.infolist():
            if re.match(r"^.*(B04|B08)(_10m|)\.jp2$", info.filename):
                zip_ref.extract(info, path=SENTINELPRODUCTS_DIR)
        zip_ref.close()


def get_files_in_path(path):
    return [f for f in listdir(path) if isfile(join(path, f))]


# calculate ndvi
def getndvi(nir_band, red_band):
    import numpy as np
    return np.nan_to_num((nir_band.astype(rasterio.float32) - red_band.astype(rasterio.float32)) / (nir_band + red_band))


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

def create_ndvi_rois():
    if os.path.exists(NDVI_DIR):
        shutil.rmtree(NDVI_DIR)
    os.mkdir(NDVI_DIR)
    
    features = read_geojson(REGION_DATA_FILE)
    
    for feature in features.features:
        import fiona
        geom = geopandas.GeoDataFrame.from_features(FeatureCollection([feature]))
    
        # use WGS84
        geom.crs = fiona.crs.from_epsg(4326)
    
        product = feature.properties["sentinelproduct"]
        # counter for rois per products
        i = 0
        for root, dir_names, file_names in os.walk(os.path.join(SENTINELPRODUCTS_DIR, product)):
            sorted_files = sorted(fnmatch.filter(file_names, "*.jp2"))
            if len(sorted_files) == 0:
                continue;
            
            b4, b8 = list(map(lambda item: os.path.join(root, item), sorted_files))

            with rasterio.open(b4) as red:
                red_bb = create_bb_data_frame(red.bounds.left, red.bounds.bottom, red.bounds.right, red.bounds.top)

                # project feature to target (sentinel product) crs
                projected_geom = geom.to_crs(red.crs)
                
                # ensure ROI is inside product
                if not red_bb.contains(projected_geom).bool():
                    continue

                # create a GeoJSON-like dict - needed for rasterio.mask
                roi_polygons = list(map(lambda item: json.loads(geopandas.GeoSeries(item).to_json())["features"][0]["geometry"],
                                    projected_geom.geometry))
                roi_bb = create_bb_data_frame(projected_geom.bounds.minx, projected_geom.bounds.miny,
                                              projected_geom.bounds.maxx, projected_geom.bounds.maxy)
                roi_bb_polygons = list(map(lambda item: json.loads(geopandas.GeoSeries(item).to_json())["features"][0]["geometry"],
                                                    roi_bb.geometry))

                # get mask from feature data
                red_mask, red_transform = mask.mask(red, roi_polygons, crop=True, filled=False)
                red_bb_mask, red_bb_transform = mask.mask(red, roi_bb_polygons, crop=True)


                profile = red.meta.copy()

            with rasterio.open(b8) as nir:
                # nir_cropped_mask, _ = mask.mask(nir, roi_polygons, crop=True, filled=False)
                nir_bb_mask, _ = mask.mask(nir, roi_bb_polygons, crop=True)

            ndvi = getndvi(nir_bb_mask, red_bb_mask)

            profile.update({"driver":    "GTiff",
                            "dtype":     rasterio.float32,
                            "height":    red_bb_mask.shape[1],
                            "width":     red_bb_mask.shape[2],
                            "transform": red_bb_transform})

            ndvi_file = os.path.join(NDVI_DIR, feature.properties["id"] + ".tif")
            mask_file = os.path.join(NDVI_DIR, feature.properties["id"] + ".mask")
            
            with rasterio.open(ndvi_file, "w", **profile) as dst:
                dst.write(ndvi)                
            # save mask values for later use
            red_mask.mask.dump(mask_file)

    
def populate_set(train_set, train_dir):
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.mkdir(train_dir)
    
    for feature in train_set:
        # move ndvi to train or val folder
        id = feature.properties["id"]
        src_ndvi_file = os.path.join(NDVI_DIR, id + ".tif")
        src_ndvi_mask_file = os.path.join(NDVI_DIR, id + ".mask")
        
        dest_ndvi_file = os.path.join(train_dir, id + ".tif")
        dest_ndvi_mask_file = os.path.join(train_dir, id + ".mask")
        
        
        os.rename(src_ndvi_file, dest_ndvi_file)
        os.rename(src_ndvi_mask_file, dest_ndvi_mask_file)
    
    with open(os.path.join(train_dir, "regions.geojson"), "w") as f:
        json.dump(FeatureCollection(train_set), f)


def create_training_sets():
    from sklearn.model_selection import train_test_split
    
    features = read_geojson(REGION_DATA_FILE).features
    train_set, val_set = train_test_split(features, test_size=0.2, random_state=42)
    
    populate_set(train_set, TRAIN_DIR)
    populate_set(val_set, VAL_DIR)


if os.path.exists(REGION_DATA_FILE):
    os.remove(REGION_DATA_FILE)
    
files = get_files_in_path(GEOJSON_DIR)


for file in files:
    if file.endswith(".geojson"):
        geojson_file = os.path.join(GEOJSON_DIR, file)
        download_sentinel_products_for_ROI(geojson_file)

unzip_sentinel_products()
 
create_ndvi_rois()

create_training_sets()

