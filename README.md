# Glacier Deep Inversion
A project to enable the inversion of geophysical parameters of glaciers using Deep Learning methods.

---

## Welcome

This project is free to use. To get started, you’ll need:
- **Python 3** and the libraries listed in `src/importation`.
- A text editor to create and edit JSON files.

---

## Acknowledgements

This work was funded by **ANR**, extensively tested, and executed on the **MUST platform**. For synthetic volcanic deformation based on the **Mogi model**, the Python code was adapted from a MATLAB script.

---

## Tutorial

To use this project, you’ll follow three main steps:
1. **Define your training process.**
2. **Run the training process.**
3. **Compute testing results.**

---

### 1. Define Your Training Process

This step involves writing several JSON files to specify configurations for training.

#### Dataset Description

Datasets should be saved in the `data/datasets` directory. As of now, the following variables must be included:

```json
{
  "type": "string",           // Currently, "generated" is the only implemented type.
  "generation_type": "string", // Refer to the `generators_table` variable in `src/simulations/generators.py`.
  "fixed": "dictionary",       // Specify constant values for variables used in the selected generation type.
  "varyings": "dictionary",    // Define QUANTILES intervals for variables in the test dataset.
  "distributions": "dictionary" // Specify distributions for the varying variables (see `distributions_table` in `src/distributions.py`).
}
```

#### Model Description

To build your model, you are currently limited to modules listed in the `modules_table` variable in `src/modules.py`. Guidelines are provided to help expand this functionality if needed. Once familiar, write your model’s configuration as a JSON file and save it in `data/models`. Each application requires its own model description.

The following variables must be defined:

```json
{
  "inputs": "dictionary",      // Input variables with parameters like size, etc.
  "outputs": "list",           // Names of output variables.
  "operations": "dictionary",  // Modules to use with their respective parameters.
  "branches": "list"           // Define branches in the model (e.g., splits for predictions or multi-input models). Use `x_name` and `y_name` as IDs for input and output respectively.
}
```

#### Training Description

Save training configurations in `experiences/defined_experiences`. The JSON structure is as follows:

```json
{
  "model": "string",             // Name of the JSON file in `data/models`.
  "dataset": "string",           // Name of the dataset file in `data/datasets`.
  "inputs": "list",              // List of inputs to use.
  "outputs": "list",             // List of outputs to use.
  "split": "list",               // Proportions for train and validation datasets.
  "sizes": "list",               // Number of samples for training, validation, and testing per epoch.
  "epochs": "integer",           // Number of training epochs.
  "batch_size": "integer",       // Batch size.
  "optimizer": "string",         // Name of the optimizer.
  "learning_rate": "float",      // Learning rate.
  "additional_callbacks": "list", // Callbacks to use (refer to `callbacks_dict` in `src/experiment.py`).
  "losses": "dictionary",        // Losses to use (refer to `losses_dict` in `src/experiment.py`).
  "metrics": "dictionary"        // Metrics to use (refer to `metrics_dict` in `src/experiment.py`).
}
```

---

### 2. Run the Training Process

This step trains your model.

Once all necessary files are ready, execute `run_training.py` with the following parameters:

- `--json_data`: *(string)* Name of the file in `experiences/defined_experiences`.
- `--seed`: *(float)* A number (cast to an integer) to define random operations.
- `--storage_path`: *(string)* Directory where results will be saved.
- `--pretrain`: *(string)* Leave empty (this feature is not yet functional).

> **Note:** Monitor your system’s memory usage. A high number of epochs can generate large files.

---

### 3. Compute Testing Results

This feature estimates parameters with the trained model. **However, manual edits to the Python script are currently required.** 

#### Current Limitations
- This function only supports estimating `depth` and `delta_volume` variables corresponding to the Mogi model.
- Estimations are made on a 2D grid of resolution **128x128**, with ranges:
  - `depth`: 0 to 4 meters.
  - `delta_volume`: 0 to \(4 \times 10^6 \, \text{m}^3\).
- To modify variables or ranges, edit the `distributions` variable in `src/infer_matrix.py:infer_matrix()`.

#### Running the Estimation

Execute `run_matrixing.py` with the following parameters:

- `--training_run`: *(string)* Name of your run. If issues arise, edit the `model_path` variable in `src/infer_matrix.py:infer_matrix()`.
- `--dataset`: *(string)* Name of your dataset JSON file.
- `--seed`: *(float)* Used to specify the iteration for matrix computation (cast to an integer).
- `--storage_path`: *(string)* Directory where the matrix will be saved.

The resulting matrix will be saved and ready for interpretation.
