from functions.importation import json, os, tensorflow as tf
from drawing.predict import predict_map, predict_profiles

def draw(parameters):
    """
    for model_name (if several models build differently, like ResNet and VGGNet)

        load model

        for seed (different training start)
            for epoch (each step of the training)

                load weights

                for profile (each profile you want to test)

                    do prediction

                for feature (each polygon you want to test)

                    do prediction
    """
    model_name = parameters["model_name"]
    
    drawing_parameters = parameters["drawing_description"]

    features = drawing_parameters["features"]
    profiles_names = drawing_parameters["profiles"]

    model_path = os.path.join(
        "deployed",
        "models",
        model_name,
    )

    with open(os.path.join(model_path, 'construction_parameters.json')) as file:
        construction_parameters = json.load(file)
    model = tf.keras.models.load_model(model_path)

    seeds_path = os.path.join(model_path, "seeds")
    seeds = os.listdir(seeds_path)

    for seed in seeds:

        weights_paths = os.path.join(seeds_path, seed, "weights")
        epochs = os.listdir(weights_paths) # epochs are the weights file

        for epoch in epochs:

            weights_path = os.path.join(weights_paths, epoch)
            model.load_weights(weights_path)

            save_location = os.path.join(
                "figures",
                model_name,
                seed,
                epoch.split(".")[0]
            )

            if os.path.exists(save_location):
                continue
            
            if epoch.split(".")[0] == "best":
                for feature in features:
                    predict_map(
                        model,
                        parameters,
                        construction_parameters,
                        feature,
                        save_location
                    )

            else:
                """"
                for profiles_name in profiles_names:
                    predict_profiles(
                        model,
                        parameters,
                        construction_parameters,
                        profiles_name,
                        save_location
                    )
                """