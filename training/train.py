from functions.importation import os, keras, tensorflow as tf, json, datetime, numpy as np
from functions.usual_functions import exist_directory
from training.load_dataset import load_dataset
from training.create_model import create_model
from training.create_dataset import create_dataset

class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, file_path):
        super().__init__()
        self.validation_data = validation_data
        self.file_path = file_path

    def on_epoch_end(self, epoch, logs=None):

        matrix = [[0,0],[0,0]]

        for x, y in self.validation_data:
            y_pred = self.model.predict(x)
            for y_i, y_pred_i in zip(list(y["ice_classification"]), list(y_pred["ice_classification"])):
                matrix[int(y_i>.5)][int(y_pred_i>.5)] += 1
        with open(self.file_path, 'a') as f:
            f.write(f"Epoch {epoch+1} confusion matrix:\n{matrix}\n")

def nan_mae(y_true, y_pred):
    y_pred_corrected = tf.boolean_mask(y_pred, ~tf.math.is_nan(y_true))
    y_true_corrected = tf.boolean_mask(y_true, ~tf.math.is_nan(y_true))
    loss = tf.keras.losses.mae(y_true_corrected, y_pred_corrected)
    return loss if not tf.math.is_nan(loss) else 0.

def nan_mse(y_true, y_pred):
    y_pred_corrected = tf.boolean_mask(y_pred, ~tf.math.is_nan(y_true))
    y_true_corrected = tf.boolean_mask(y_true, ~tf.math.is_nan(y_true))
    loss = tf.keras.losses.mse(y_true_corrected, y_pred_corrected)
    return loss if not tf.math.is_nan(loss) else 0.

def nan_binary_crossentropy(y_true, y_pred):
    y_pred_corrected = tf.boolean_mask(y_pred, ~tf.math.is_nan(y_true))
    y_true_corrected = tf.boolean_mask(y_true, ~tf.math.is_nan(y_true))
    loss = tf.keras.losses.binary_crossentropy(y_true_corrected, y_pred_corrected)
    return loss if not tf.math.is_nan(loss) else 0.

def nan_binary_accuracy(y_true, y_pred):
    y_pred_corrected = tf.boolean_mask(y_pred, ~tf.math.is_nan(y_true))
    y_true_corrected = tf.boolean_mask(y_true, ~tf.math.is_nan(y_true))
    loss = tf.keras.metrics.binary_accuracy(y_true_corrected, y_pred_corrected)
    return loss if not tf.math.is_nan(loss) else 0.

losses_def = {
    "nan_mae": nan_mae,
    "nan_mse": nan_mse,
    "mae": tf.keras.losses.MeanAbsoluteError(name = "mae"),
    "mse": tf.keras.losses.MeanSquaredError(name = "mse"),
    "binary_crossentropy": tf.keras.losses.BinaryCrossentropy(name = "binary_crossentropy"),
    "nan_binary_crossentropy": nan_binary_crossentropy,
}

metrics_def = {
    "nan_mae": nan_mae,
    "nan_mse": nan_mse,
    "mae": tf.keras.losses.MeanAbsoluteError(name = "mae"),
    "mse": tf.keras.losses.MeanSquaredError(name = "mse"),
    "binary_crossentropy": tf.keras.metrics.BinaryCrossentropy(name = "binary_crossentropy"),
    "binary_accuracy": tf.keras.metrics.BinaryAccuracy(name = "binary_accuracy"),
    "nan_binary_crossentropy": nan_binary_crossentropy,
    "nan_binary_accuracy": nan_binary_accuracy
}

optimizers_def = {
    "adam": tf.keras.optimizers.Adam
}

def train(
    parameters: dict,
) -> None: 

    dataset_parameters = parameters["dataset_description"]

    model_path = os.path.join(
        "deployed",
        "models",
        parameters["model_name"]
    )
    if not os.path.exists(model_path):
        create_model(parameters)

    dataset_path = os.path.join(
        "deployed",
        "datasets",
        dataset_parameters["dataset_name"]
    )
    if not os.path.exists(dataset_path):
        create_dataset(
            name = dataset_parameters["dataset_name"],
            shapefile_file = dataset_parameters["shapefile_file"],
            shapefile_fieldname = dataset_parameters["shapefile_fieldname"],
            rasters = dataset_parameters["rasters"],
            epsg = dataset_parameters["epsg"],
            buffer_distance = dataset_parameters["buffer_distance"]
        )

    dataset_name = dataset_parameters["dataset_name"]
    optimizer_name = parameters["optimizer_name"]
    epochs = parameters["epochs"]
    batch_size = parameters["batch_size"]
    criterion = parameters["criterion"]
    losses_name = parameters["losses_name"]
    metrics_name = parameters["metrics_name"]
    seeds_number = parameters["seeds_number"]
    
    with open(os.path.join(model_path, "model_parameters.json"), "w") as file:
        json.dump(parameters, file)
        
    with open(os.path.join(model_path, "construction_parameters.json"), "r") as file:
        construction_parameters = json.load(file)

    inputs = construction_parameters["inputs"]
    outputs = construction_parameters["outputs"]
    
    losses = {
        output["name"]: losses_def[losses_name[output["name"]]] for output in outputs
    }

    metrics = {
        output["name"]: [
            metrics_def[metric_name] for metric_name in metrics_name[output["name"]]
        ] for output in outputs
    }
    
    dataset_path = os.path.join(
        "deployed",
        "datasets",
        dataset_name,
    )

    for i in range(seeds_number):
        tf.keras.backend.clear_session()
        seed = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        tf.random.set_seed(i)
        np.random.seed(i)
        
        optimizer = optimizers_def[optimizer_name]
        
        to_split_dataset = load_dataset(
            dataset_path = dataset_path,
            inputs = inputs,
            outputs = outputs,
        )

        train_dataset = to_split_dataset["train"]
        validation_dataset = to_split_dataset["validation"]
        test_dataset = to_split_dataset["test"]

        def data_augmentation(img, rng):
            # Apply data augmentation
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.rot90(img, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
            return img

        train_dataset.shuffle(1024).batch(batch_size).shuffle(128,reshuffle_each_iteration = True).prefetch(1)

        validation_dataset.batch(batch_size).prefetch(1)
        test_dataset.batch(batch_size).prefetch(1)

        create_model(parameters) # THE MODEL IS FULLY RECONSTRUCTED (for weight initialization) ALONG WITH FOLDER CREATION ETC. NEEDS TO CHANGE !!!
        model = tf.keras.models.load_model(model_path)

        model.compile(
            loss = losses,
            metrics = metrics,
            optimizer = optimizer(),
        )

        train_dataset = train_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        train_path = os.path.join(
            model_path,
            "seeds",
            seed
        )

        logs_dir = os.path.join(train_path, "logs")  # where logs are saved
        weights_dir = os.path.join(train_path, "weights")  # where models weights are saved
        exist_directory(logs_dir)  # make the directory if it dont exist (see exist_directory documentation)
        exist_directory(weights_dir)

        tensorboard_callback = keras.callbacks.TensorBoard(  # callback to save logs
            log_dir = logs_dir
        )

        callback_path = os.path.join(weights_dir, "epoch_{epoch:02d}.h5")  # model weights path update with each epoch
        callback = keras.callbacks.ModelCheckpoint(  # callback to save model weights at each epoch
            filepath = callback_path,
            save_weights_only = True,
            verbose = 2
        )

        best_callback_path = os.path.join(weights_dir, "best.h5")
        best_callback = keras.callbacks.ModelCheckpoint(  # callback to save model weights at best epoch
            filepath = best_callback_path,
            monitor = criterion,  # what metric decide if the model is good
            save_weights_only = True,
            save_best_only = True,
            verbose = 2
        )
        
        earlystopping_callback = keras.callbacks.EarlyStopping(
            monitor = criterion,
            patience = 16,
            restore_best_weights = True
        )

        reduceLROnPlateau_callback = keras.callbacks.ReduceLROnPlateau(
            monitor = criterion,
            factor = 0.1,
            patience = 4,
            min_lr = 1e-6,
            verbose = True
        )

        confusionMatrix_callback = ConfusionMatrixCallback(
            validation_dataset, 
            os.path.join(train_path, "confusion_matrix.txt")
        )

        callbacks = [
            callback,
            best_callback,
            #earlystopping_callback,
            reduceLROnPlateau_callback,
            tensorboard_callback,
            #confusionMatrix_callback
        ]

        model.fit(
            x = train_dataset,
            epochs = epochs,
            batch_size = batch_size,
            shuffle = True,  # shuffle dataset at each epoch
            workers = 2,
            use_multiprocessing = True,
            validation_data = validation_dataset,
            
            callbacks = callbacks,
            verbose = 2
        )

        del model
