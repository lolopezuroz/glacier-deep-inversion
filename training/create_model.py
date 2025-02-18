from functions.importation import os, keras, json
from functions.usual_functions import exist_directory
from training.architectures.resnet_model import resnet_modules
from training.architectures.vggnet_model import vggnet_modules

def create_model(
    model_parameters: dict
) -> None:
    """
    create a model according to training parameters and the model
    description
    """
    model_name = model_parameters["model_name"]
    architecture = model_parameters["model_architecture"]
    area_size = model_parameters["area_size"]
    inputs = model_parameters["inputs"]
    outputs = model_parameters["outputs"]
    model_description = model_parameters["model_description"]

    if architecture == "vggnet":
        model_modules = vggnet_modules
    elif architecture == "resnet":
        model_modules = resnet_modules

    def keras_operations_constructor(
        operations: list,
        keras_input
    ):
        x = keras_input
        for operation in operations:
            x = model_modules(operation, x)
        return x

    main_operations = [
        operation for operation in model_description["main"]["main"]["operations"]
    ]

    output_operations = [
        operation for operation in list(model_description["outputs"].values())[0]["operations"]
    ]
    
    keras_inputs = {}
    processing_inputs = []
    for input in inputs:
        
        tile_size = int(area_size // input["resolution"])

        input_operations = model_description["inputs"][input["name"]]["operations"]

        keras_input = keras.Input(
            shape = (tile_size, tile_size, 1,),
            name = input["name"]
        )
        
        input["tile_size"] = tile_size
        input["scalar"] = False

        keras_inputs[input["name"]] = keras_input

        processing_inputs.append(keras_operations_constructor(
            input_operations,
            keras_input
        ))

    if len(processing_inputs) == 1:
        processing_inputs = processing_inputs[0]
    y = keras_operations_constructor(
        main_operations,
        processing_inputs,
    )
    
    keras_outputs = {}
    for output in outputs:

        output["tile_size"] = int(area_size // output["resolution"])

        output_operations = model_description["outputs"][output["name"]]["operations"]

        keras_outputs[output["name"]] = keras_operations_constructor(
            output_operations,
            y,
        )

    model = keras.Model(
        inputs = keras_inputs,
        outputs = keras_outputs,
        name = model_name,
    )

    model_directory = os.path.join( # generate unique directory to save train states and parameters
        "./",
        "deployed",
        "models",
        model_name,
    )
    
    exist_directory(model_directory)

    # save model graph
    #keras.utils.plot_model(
    #    model,
    #    os.path.join(model_directory, f"{model_name}.png"),
    #    show_shapes = True
    #)

    construction_parameters = {
        "name": model_name,
        "area_size": area_size,
        "inputs": inputs,
        "outputs": outputs,
        "model_parameters": model_parameters,
    }

    print(model.summary())

    # save model construction parameters
    with open(os.path.join(
        model_directory,
        "construction_parameters.json",
    ), "w") as f:
        json.dump(construction_parameters, f)
    
    # save model
    model.save(model_directory)
