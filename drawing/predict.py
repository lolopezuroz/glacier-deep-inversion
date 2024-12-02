from functions.importation import os, numpy as np, gdal, convert_to_tensor, json, ogr
from functions.usual_functions import exist_directory, find_ref, extract_profile, extract_samples
import time

def predict_profiles(
    model,
    model_parameters: dict,
    construction_parameters: dict,
    profiles_name: str,
    save_location: str,
) -> None:

    inputs = construction_parameters["inputs"]
    outputs = construction_parameters["outputs"]

    outputs_name = [output["name"] for output in outputs]
    inputs_name = [input["name"] for input in inputs]

    linear_outputs = ["linear_ice_thickness"]

    profile_path = os.path.join(
        "drawing",
        "profiles",
        f"{profiles_name}.shp"
    )
    
    dataset_path = feature_path = os.path.join(
        "../",
        "deployed",
        "datasets",
        model_parameters["dataset_description"]["dataset_name"]
    )
    for affiliation in ["train", "test", "validation"]:
        feature_path = os.path.join(dataset_path, affiliation, profiles_name)
        if os.path.exists(feature_path):
            break

    save_profiles = os.path.join(save_location, profiles_name)  # where to save the images
    exist_directory(save_profiles)  # create directory if don't exist

    driver = ogr.GetDriverByName("ESRI Shapefile")
    shapefile = driver.Open(profile_path)

    profiles = shapefile.GetLayer()

    for i, profile in enumerate(profiles):

        samples = extract_profile(
            profile = profile.GetGeometryRef(),
            feature_path = feature_path,
            products = inputs + outputs,
        )

        linear_samples = extract_profile(
            profile = profile.GetGeometryRef(),
            feature_path = feature_path,
            products = linear_outputs,
        )

        n = len(list(samples.values())[0])
        samples_2 = [{key: convert_to_tensor(values[i]) for key, values in samples.items() if key in inputs_name} for i in range(n)]

        predictions = {output_name: [] for output_name in outputs_name}

        for sample in samples_2:
            prediction = model(sample)
            for output_name in outputs_name:
                predictions[output_name].append(float(prediction[output_name]))

        with open(os.path.join(
            save_profiles,
            f"{profiles_name}_profile_{i}.json"
        ), "w") as file:
            json.dump(predictions, file)

        reference_data_path = os.path.join(
            save_profiles,
            f"{profiles_name}_reference_{i}.json"
        )

        if not os.path.exists(reference_data_path):
            samples_dump = {key: [float(value) for value in samples[key]] for key in outputs_name}
            with open(reference_data_path, "w") as file:
                json.dump(samples_dump, file)

def predict_map(
    model,
    model_parameters,
    construction_parameters,
    feature_name: str,
    save_location: str,
) -> None:
    """
    """

    inputs = construction_parameters["inputs"]
    outputs = construction_parameters["outputs"]
    area_size = model_parameters["area_size"]
    
    outputs_name = [output["name"] for output in outputs]

    dataset_path = feature_path = os.path.join(
        "../deployed",
        "datasets",
        model_parameters["dataset_description"]["dataset_name"]
    )
    for affiliation in ["train", "test", "validation"]:
        feature_path = os.path.join(dataset_path, affiliation, feature_name)
        if os.path.exists(feature_path):
            break

    samples = extract_samples(
        feature_path,
        inputs,
        1.,
    )

    n = len(list(samples.values())[0])
    samples_2 = [{key: values[i] for key, values in samples.items()} for i in range(n)]

    predictions = {output_name: [] for output_name in outputs_name}

    for sample in samples_2:
        prediction = model(sample)
        for output_name in outputs_name:
            predictions[output_name].append(prediction[output_name])
    
    ref_product = find_ref(inputs)
    image_path = os.path.join(feature_path, ref_product["name"]) + ".tif"
    raster = gdal.Open(image_path)

    tile_size = int(area_size / ref_product["resolution"])

    columns = raster.RasterXSize
    rows = raster.RasterYSize
    
    for key, value in predictions.items():
        predictions[key] = np.array(value).reshape(
            rows - tile_size - 1,
            columns - tile_size - 1
        )

    driver = gdal.GetDriverByName('GTiff')

    geo_transform = list(raster.GetGeoTransform())

    geo_transform[0] += ref_product["resolution"] * (tile_size) / 2
    geo_transform[3] -= ref_product["resolution"] * (tile_size) / 2

    save_images = os.path.join(save_location, feature_name)  # where to save the images
    exist_directory(save_images)  # create directory if don't exist

    for key in predictions.keys():
        out_ds = driver.Create(
            os.path.join(save_images, key + "_prediction.tif"),
            columns - 4,
            rows - 4,
            1,
            gdal.GDT_Float32
        )
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(predictions[key])
        out_ds.SetGeoTransform(geo_transform)
        out_ds.SetProjection(raster.GetProjection())
        out_band.FlushCache()