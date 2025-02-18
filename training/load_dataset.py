from functions.importation import tensorflow as tf, os, math, Image, numpy as np
from functions.usual_functions import find_ref

def load_dataset(
    dataset_path: str,
    inputs: list,
    outputs: list,
) -> dict:
    """
    prepare a tensorflow datasets

    dataset_path: from where to load the data
    features_attributions: at which correspond each feature (train, validation, test, skip)
    inputs:
    outputs:

    return dataset split into train, validation and test
    """
    products = inputs+outputs
    inputs_name = [input["name"] for input in inputs]
    outputs_name = [output["name"] for output in outputs]

    def decoder(sample: dict) -> dict:
        """
        function that yield dataset samples

        sample:dict product name as key and value (scalar or tensor) as entry

        return:tuple
        """
        xs = {}
        ys = {}
        for name, value in sample.items():
            if name in inputs_name:
                xs[name] = value
            if name in outputs_name:
                ys[name] = value
        return (xs, ys)

    def build_dataset(
        glacier_folder: str,
        overlap: float, 
        shuffle: bool
    ):

        def load_img(feature_path, product_name):
            array_path = feature_path + f"/{product_name}.tif"
            return tf.py_function(
                lambda filename : np.array(Image.open(str(filename.numpy())[2:-1])),
                [array_path],
                tf.float32
            )

        def get_glacier_crops_bbox_dataset(ref_image, ref_tile_size):
            spacing = math.ceil((1 - overlap) * ref_tile_size) if overlap < 1. else 1
            xn = tf.shape(ref_image)[0] - ref_tile_size - 1
            yn = tf.shape(ref_image)[1] - ref_tile_size - 1

            xs, ys = tf.meshgrid(
                tf.range(0, xn, spacing, dtype=tf.int32),
                tf.range(0, yn, spacing, dtype=tf.int32)
            )
            xs = tf.reshape(xs, [-1])
            ys = tf.reshape(ys, [-1])
            boxes_raw = tf.stack([xs, ys], axis=1) #passer par un tableau 2D, plus compact/efficace
            return tf.data.Dataset.from_tensor_slices(boxes_raw)

        def crop_scalar(image, coordinates, shift, tile_size):
            x = coordinates[0]
            y = coordinates[1]
            extract = image[
                x * shift + tile_size//2: x * shift + tile_size//2+2,
                y * shift + tile_size//2: y * shift + tile_size//2+2
            ]
            return tf.expand_dims(tf.math.reduce_mean(extract), axis=-1)

        def crop_image(image, coordinates, shift, tile_size):
            
            x = coordinates[0]
            y = coordinates[1]
            extract = image[
                x * shift: x * shift + tile_size,
                y * shift: y * shift + tile_size,
            ]

            extract = tf.where(tf.math.is_nan(extract), 0., extract)
            
            return tf.expand_dims(extract, axis=-1)

        def extract_samples(single_glacier_path):

            #load all products a single time
            #get the lowest resolution product
            ref_product = find_ref(products)
            ref_tile_size = ref_product["tile_size"]

            for product in products:
                product["shift"] = product["tile_size"] // ref_tile_size
                product["image"] = load_img(single_glacier_path, product["name"])
            ref_image = ref_product["image"]

            #prepare crop coordinates on the lowest resolution product (ref_image)
            crop_coord_dataset=get_glacier_crops_bbox_dataset(ref_image, ref_tile_size)
                
            @tf.function #will unroll the loop :o)
            def get_single_crop(sample_coord):
                # generate a dictionnary that gathers all products falling into a single crop
                sample={}
                for product in products:
                    if product['scalar']:
                        sample[product["name"]] = crop_scalar(product["image"], sample_coord, product["shift"], product["tile_size"])
                    else:
                        sample[product["name"]]= crop_image(product["image"], sample_coord, product["shift"], product["tile_size"])
                return sample                
            
            #extract all crops and get a dictionnary of product for each of them
            crops_dataset=crop_coord_dataset.map(get_single_crop)
            return crops_dataset

        glacier_paths = [os.path.join(glacier_folder, glacier) for glacier in os.listdir(glacier_folder)]
        
        glacier_dataset = tf.data.Dataset.from_tensor_slices(glacier_paths)
        dataset = glacier_dataset.interleave(
            map_func = extract_samples, 
            cycle_length = 4,
            block_length = 16, 
            deterministic = not(shuffle)
        )
        filter_outside = False
        if filter_outside:
            my_filter = lambda x: tf.math.is_nan(x["ice_thickness"])
            dataset = dataset.filter(my_filter)

        return dataset.map(decoder)
    
    train_dataset = build_dataset(os.path.join(dataset_path, "train"), .1, True)
    validation_dataset = build_dataset(os.path.join(dataset_path, "validation"), .1, False)
    test_dataset = build_dataset(os.path.join(dataset_path, "test"), .1, False)

    i = 0
    for v in train_dataset: i += 1
    j = 0
    for v in validation_dataset: j += 1
    k = 0
    for v in test_dataset: k += 1

    print(i, j, k)
    
    return {
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
    }
