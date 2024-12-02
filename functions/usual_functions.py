from functions.importation import os, numpy as np, Image, math, gdal, shapely_wkt

def exist_directory(path:str) -> str:
    """
    check if "path" represent an existent directory
    if not, it create the full path to it

    path:str

    return:str the path of tested directory
    """
    if not os.path.isdir(path):
        indice = path.rfind("/")
        if indice == -1:
            return
        parent_directory = path[:indice]
        exist_directory(parent_directory)
        os.mkdir(path)
    return path

def find_ref(products:list) -> dict:
    ref_index = np.argmax([product["resolution"] for product in products])
    return products[ref_index]

def extract_profile(
    profile,
    feature_path: str,
    products: list,
) -> dict:
    """
    get tiles from rasters within a folder
    """

    ref_product = find_ref(products)

    ref_tile_size = ref_product["tile_size"]
    
    ref_array_path = os.path.join(
        feature_path,
        f"{ref_product['name']}.tif"
    )

    raster = gdal.Open(ref_array_path)
    transform = raster.GetGeoTransform()

    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]

    line = shapely_wkt.loads(profile.ExportToWkt())

    distances = np.arange(0, line.length, ref_product["resolution"])
    points = [line.interpolate(distance) for distance in distances]

    pixels_list = [
        (
            int((yOrigin - point.y) / pixelHeight - ref_tile_size / 2),
            int((point.x - xOrigin) / pixelWidth - ref_tile_size / 2)
        ) for point in points
    ]

    samples = {product["name"]:[] for product in products}
    
    for product in products:
        
        tile_size = product["tile_size"]

        name = product["name"]
        array_path = os.path.join(feature_path, f"{name}.tif") 
        array = np.array(Image.open(array_path))
        
        shift = int(product["tile_size"] // ref_tile_size)
        
        if product["scalar"]:
            for x, y in pixels_list:
                extract = array[
                    x * shift + tile_size // 2: x * shift + tile_size // 2 + 2,
                    y * shift + tile_size // 2: y * shift + tile_size // 2 + 2
                ]

                samples[name].append(np.expand_dims(np.mean(extract), axis=0))
        else:
            for x, y in pixels_list:

                tile = array[
                    x * shift : x * shift + tile_size,
                    y * shift : y * shift + tile_size,
                ]

                tile = np.expand_dims(tile, axis = 0)
                tile = np.nan_to_num(tile)

                samples[name].append(np.copy(tile))

    return samples

def extract_samples(
    feature_path: str, 
    products: list, 
    overlap: float,
    select_proportion: float = 1.,
    random_shift: bool = False,
) -> dict:
    """
    get tiles from rasters within a folder
    """

    ref_product = find_ref(products)

    ref_tile_size = ref_product["tile_size"]
    
    ref_array_path = os.path.join(
        feature_path,
        f"{ref_product['name']}.tif"
    )
    ref_array = np.array(Image.open(ref_array_path))
    
    spacing = math.ceil((1 - overlap) * ref_tile_size) if overlap < 1. else 1

    xn, yn = np.array(np.shape(ref_array)) - ref_tile_size - 1
    del ref_array

    xs, ys = np.indices((xn, yn))
    
    xs = xs[::spacing, ::spacing]
    
    ys = ys[::spacing, ::spacing]

    xs = xs.flatten()
    ys = ys.flatten()

    if select_proportion < 1.:
        to_select = np.arange(len(xs))
        np.random.shuffle(to_select)
        to_select = to_select[:int(len(xs) * select_proportion)]

        xs = xs[to_select]
        ys = ys[to_select]

        del to_select

    samples = {product["name"]:[] for product in products}
    
    for product in products:
        
        tile_size = product["tile_size"]

        name = product["name"]
        array_path = os.path.join(feature_path, f"{name}.tif") 
        array = np.array(Image.open(array_path))
        
        shift = int(product["tile_size"] // ref_tile_size)
        
        if product["scalar"]:
            for x, y in zip(xs, ys):
                extract = array[
                    x * shift + tile_size // 2: x * shift + 2 + tile_size // 2,
                    y * shift + tile_size // 2: y * shift + 2 + tile_size // 2
                ]

                samples[name].append(np.expand_dims(np.mean(extract), axis=0))
        else:
            for x, y in zip(xs, ys):

                tile = array[
                    x * shift: x * shift + tile_size,
                    y * shift: y * shift + tile_size,
                ]

                tile = np.expand_dims(tile, axis = 0)
                tile = np.nan_to_num(tile)

                samples[name].append(np.copy(tile))

    return samples

def array_to_raster(dst_filename, array, old_dataset):

    driver = gdal.GetDriverByName('GTiff')

    y_pixels, x_pixels = np.shape(array)

    dataset = driver.Create(
        dst_filename,
        x_pixels,
        y_pixels,
        1,
        gdal.GDT_Float32
    )

    dataset.SetGeoTransform(old_dataset.GetGeoTransform())

    dataset.SetProjection(old_dataset.GetProjection())
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()
    return dataset, dataset.GetRasterBand(1)