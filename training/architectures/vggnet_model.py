from functions.importation import keras, Tensor

def vggnet_convolutions(
    x: Tensor,
    filters: list,
) -> Tensor:
    """
    the convolutional part of a vggnet

    x: Tensor
    filters: list[int] number of dimensions used in each convolution
    strides: list[int]

    return: Tensor
    """
    for filter in filters:
        layer = keras.layers.Conv2D(
            filters = filter,
            kernel_size = 3,
            strides = 1,
            padding = "valid",
            activation = "relu",
            kernel_regularizer = "l2"
        )
        x = layer(x)
    return x

def vggnet_denses(
    x: Tensor,
    denses_filters: list,
    last_activation: str = "relu"
) -> Tensor:
    """
    the denses of vggnet employed before exiting the model

    x: Tensor
    dense_filters: list[int]
    last_activation: str what type of activation to use before exiting model (relu by default)

    return: Tensor
    """
    x = keras.layers.Flatten()(x)  # turn 3D tensor into 1D tensor
    for i, dense_filter in enumerate(denses_filters):
        if i+1 != len(denses_filters):
            activation = "relu"
            use_bias = True
        else:
            activation = last_activation  # to adapt to product type (regression or classification) last activation can depend
            use_bias = False  # no default value
        layer = keras.layers.Dense(
            units = dense_filter,
            activation = activation,
            use_bias = use_bias
        )
        x = layer(x)
    return x

def vggnet_modules(operation, x):

    if operation["name"] == "convolutions":
        return vggnet_convolutions(
            x,
            operation["filters"]
        )
    elif operation["name"] == "denses":
        return vggnet_denses(
            x,
            operation["filters"],
            operation["last_activation"]
        )
    elif operation["name"] == "upsample":
        return keras.layers.MaxPooling2D()(x)

    elif operation["name"] == "fusion":
        return keras.layers.Add()(x)