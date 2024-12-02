from functions.importation import keras, Tensor

def identity_block(x: Tensor, filters: list) -> Tensor:
    """
    the convolutional part of a resnet block

    x: Tensor
    filters: list[int] number of dimensions used in each convolution

    return: Tensor
    """
    for i, filter in enumerate(filters):
        activation = "relu" if i+1 != len(filters) else None  # no activation on last convolution (will occur later)
        layer = keras.layers.Conv2D(
            filters = filter,
            kernel_size = 3,
            strides = 1,  # cant use stride for pooling because wont accomodate with residuals
            padding = "same",
            activation = activation,
            kernel_regularizer = "l2",
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
        )
        x = layer(x)
    return x

def shortcut(
    x: Tensor,
    out_dim: int,
) -> Tensor:
    """
    the adaptation of residuals part of a resnet block

    x: Tensor
    out_dim: int the dimension which residuals needs to atteign
    cropping: int number of edges to drop due to convolution (should be equal to number of convolutions in the resnet block)

    return: Tensor
    """
    layer = keras.layers.Conv2D(
        filters = out_dim,
        kernel_size = 1,
        strides = 1,
        activation = None,  # activation will occur after addition of convolutional product and residuals
        kernel_regularizer = "l2",
        kernel_initializer = 'glorot_uniform',
        bias_initializer = 'zeros'
    )
    x = layer(x)
    return x

def resnet_block(
    x:Tensor,
    filters:list
) -> Tensor:
    """
    major component of a resnet

    x: Tensor
    filters: list[int]

    return: Tensor
    """
    x_identity = identity_block(x, filters)  # convolutional part
    x_shortcut = shortcut(x, filters[-1])  # adaptation of residuals
    x = keras.layers.Add()([x_identity, x_shortcut])  # fuse product of convolutions and residuals
    x = keras.layers.ReLU()(x)  # activation that was skipped for both

    return x

def resnet_denses(
    x: Tensor,
    denses_filters: list,
    last_activation: str = "relu"
) -> Tensor:
    """
    the denses of resnet employed before exiting the model

    x: Tensor
    dense_filters: list[int]
    last_activation: str what type of activation to use before exiting model (relu by default)

    return: Tensor
    """
    x = keras.layers.Flatten()(x)  # turn 3D tensor into 1D tensor
    for i, dense_filter in enumerate(denses_filters):
        activation = "relu" if i+1 != len(denses_filters) else last_activation  # to adapt to product type (regression or classification) last activation can depend
        layer = keras.layers.Dense(
            units = dense_filter,
            activation = activation,
            use_bias = False,
            kernel_initializer='glorot_uniform'
        )
        x = layer(x)
    return x

def resnet_modules(operation, x):

    if operation["name"] == "convolutions":
        return resnet_block(
            x,
            operation["filters"]
        )
    
    if operation["name"] == "convolution_1":
        return keras.layers.Conv2D(
            filters = operation["filter"],
            kernel_size = 1,
            strides = 1,
            activation = "relu",
            kernel_regularizer = "l2",
            kernel_initializer = 'glorot_uniform',
            bias_initializer = 'zeros',
        )(x)

    elif operation["name"] == "denses":
        return resnet_denses(
            x,
            operation["filters"],
            operation["last_activation"]
        )

    elif operation["name"] == "downsample":
        return keras.layers.MaxPooling2D()(x)

    elif operation["name"] == "addition":
        return keras.layers.Add()(x)

    elif operation["name"] == "concatenation":
        return keras.layers.Concatenate()(x)