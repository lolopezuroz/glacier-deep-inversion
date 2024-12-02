from functions.importation import math, os, gdal, numpy, json, ogr, osr
from functions.usual_functions import exist_directory

def create_dataset(
    name: str,
    shapefile_file: str,
    shapefile_fieldname: str,
    rasters: list,
    epsg: int,
    buffer_distance: float,
) -> None :
    """
    split raster images into patches adjusted to shapefile features
    
    name: dataset proper name
    shapefile_file: file name of shapefile
    shapefile_fieldname: which attribute correspond to feature's names
    rasters: list of different parameters for rasters
    epsg: which projection to use
    buffer_distance: what distance to add
    """
    
    dataset_path = os.path.join(
        "deployed",
        "datasets",
        name)
    exist_directory(dataset_path)

    max_resolution = max(raster["resolution"] for raster in rasters)

    # import shapefile
    shapefile_path = os.path.join(
        "data",
        shapefile_file
    )
    driver = ogr.GetDriverByName("ESRI Shapefile")
    shapefile = driver.Open(shapefile_path)
    layer = shapefile.GetLayer()

    summary = {
        "features_count": len(layer),
        "max_resolution": max_resolution,
        "buffer_distance": buffer_distance,
        "products": rasters,
        "epsg": epsg,
        "features": [],
    }

    # prepare reprojection
    shapefile_src = layer.GetSpatialRef()
    target_src = osr.SpatialReference()
    target_src.ImportFromEPSG(epsg)
    transform = osr.CoordinateTransformation(shapefile_src, target_src)

    train_attribution = {
        0: "train",
        1: "validation",
        2: "test"
    }
    
    # iterate over all features
    for feature in layer:

        # create feature folder
        feature_name = feature.GetField(shapefile_fieldname)
        feature_train = feature.GetField("train")
        feature_path = os.path.join(
            dataset_path,
            train_attribution[feature_train],
            feature_name,
        )
        exist_directory(feature_path)

        # get spatial extent
        feature_geometry = feature.GetGeometryRef()
        feature_geometry.Transform(transform) # reproject geometry
        feature_envelope = list(feature_geometry.GetEnvelope())
        
        # add buffer distance to extent
        feature_envelope[0] -= buffer_distance * 2
        feature_envelope[1] += buffer_distance * 2
        feature_envelope[2] -= buffer_distance * 2
        feature_envelope[3] += buffer_distance * 2

        # extent size
        length_x = feature_envelope[1] - feature_envelope[0]
        length_y = feature_envelope[3] - feature_envelope[2]

        # amount of size missing to precisely match raster crop
        residual_x = length_x % max_resolution
        residual_y = length_y % max_resolution

        # correct the envelope
        feature_envelope[0] -= residual_x/2
        feature_envelope[1] += residual_x/2
        feature_envelope[2] -= residual_y/2
        feature_envelope[3] += residual_y/2

        ref_pixel_length_x = math.ceil(length_x / max_resolution)
        ref_pixel_length_y = math.ceil(length_y / max_resolution)

        for raster in rasters:

            raster_path = os.path.join("data", raster["file"])
            raster_name = raster["name"]
            raster_resolution = raster["resolution"]

            tile_path = os.path.join(feature_path, raster_name + ".tif")

            warp_options = gdal.WarpOptions(
                targetAlignedPixels=False, # actually mess up the alignment if True (try to fit the extent weirdly)
                outputBounds=(
                    feature_envelope[0],
                    feature_envelope[2],
                    feature_envelope[1],
                    feature_envelope[3],
                ),
                width=int(ref_pixel_length_x * max_resolution // raster_resolution),
                height=int(ref_pixel_length_y * max_resolution // raster_resolution),
                dstNodata=numpy.nan,
                resampleAlg=gdal.GRA_Bilinear,
                multithread=True,
                dstSRS=target_src)

            tile = gdal.Warp(
                tile_path,
                raster_path,
                options=warp_options)
            tile.FlushCache()

        feature_summary = {
            "name": feature_name,
            "envelope": feature_envelope,
        }
        summary["features"].append(feature_summary)

    summary_path = os.path.join(dataset_path,"_summary.json")
    with open(summary_path, "w") as file:
        json.dump(summary, file)

    return dataset_path