# Glacier Deep Inversion

A project designed to enable the inversion of geophysical parameters of glaciers using Deep Learning methods.

---

## Welcome

This project is free to use. To get started, you’ll need:

- **Python 3** and the libraries listed in `functions/importation`.
- A text editor to create and edit JSON files.
- A **shapefile** (`.shp`).
- **Raster images** (`.tif`).
- **QGIS** for visualizing and creating data (optional).

---

## Tutorial

This process trains and utilizes models designed for regression of geophysical parameters using images. Currently, the predicted parameter must be a scalar, not an image. It is strongly recommended to set your output values with the `"scalar"` parameter set to `true` (see *Setting the Experiment > Training Description*). This ensures that the predicted parameter corresponds to the exact center of the image, enabling the use of the model as a sliding window interpreter during the drawing phase (see *Drawing*).

---

### Setting the Experiment

The experiment file is divided into several sections.

#### Training Description

This section defines the training parameters. Example structure:

```json
{
    "batch_size": integer,
    "epochs": integer,
    "min_learning_rate": float, // Minimum learning rate after reduction.
    "patience": integer,        // Number of epochs without improvement before reducing the learning rate.
    "optimizer_name": string,
    "criterion": string,        // Metric used to define "improvement."
    "model_name": string,
    "model_architecture": string, // Model architecture (defined in `training/architectures`).
    "area_size": float,         // Size of the image in geographical units.
    "seeds_number": integer,    // Number of models trained (and used for the drawing phase).
    "inputs": [                 // List of input dictionaries.
        {
            "name": string,      // Variable name.
            "resolution": float, // Pixel size resolution.
            "scalar": boolean    // If true, extracts the center value of the image; if false, retains it as an image.
        }
    ],
    "outputs": [                // List of output dictionaries (same structure as inputs).
        {
            "name": string,
            "resolution": float,
            "scalar": boolean
        }
    ],
    "losses_name": dictionary,
    "metrics_name": dictionary
}
```

---

#### Drawing Description

Defines the entities to estimate after training. Example structure:

```json
{
    "features": list,          // Entities from the shapefile dataset to include.
    "profiles": list,          // Specific profiles for estimation.
    "models": list             // Model names to use for inference.
}
```

---

#### Dataset Description

Specifies the dataset structure and settings. Example structure:

```json
"dataset_description": {
    "dataset_name": string,       // Name of the dataset folder.
    "epsg": integer,              // Coordinate system ID (e.g., 2056).
    "shapefile_file": string,     // Name of the shapefile in the data folder (include ".shp").
    "shapefile_fieldname": string, // Column name used as the ID for shapefile entities.
    "buffer_distance": float,     // Buffer distance around extracted glacier images.
    "rasters": [                  // List of raster dictionaries.
        {
            "name": string,        // Variable name.
            "resolution": float,   // Pixel size for resampling.
            "file": string         // Name of the raster file (include ".tif").
        }
    ]
}
```

---

#### Model Description

Defines the model’s input, main structure, and output configurations. Example structure:

```json
"model_description": {
    "inputs": dictionary,        // Input variables as keys with associated parameters.
    "main": dictionary,          // Main configuration of the model.
    "outputs": dictionary        // Output variables with parameters.
}
```

---

### Training

During training, the process will build and train a model on your dataset. This will be repeated a specified number of times, with each run using a different random seed. The seed affects the initialization of model parameters and data shuffling during training.

---

### Drawing (Inference)

After training, the system computes results for specified entities (from the shapefile and the fieldname ID). The inference process can use many models and will always use every one obtained through iteration, enabling computations of averages, standard deviations, and more. Output images are saved as `.tif` files, which can be easily visualized in QGIS.
