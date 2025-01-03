import tensorflow as tf


class ResNetBlock(tf.keras.Model):
    def __init__(self, filters, stride=1, name=None):
        """
        Creates a residual block for ResNet.

        Args:
            filters (int): Number of filters to be used in convolutional layer.
            stride (int): Stride size for convolutional layer.
            name (str): Name prefix for the layers in the block.
        """
        super().__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2D(
            filters, 3, strides=stride, padding="same", name=f"{name}_conv1"
        )
        self.bn1 = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")
        self.relu = tf.keras.layers.LeakyReLU(0.2)  # Changed to LeakyReLU
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv2")
        self.bn2 = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")

        if stride != 1 or filters != 64:
            self.shortcut = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(filters, 1, strides=stride),
                    tf.keras.layers.BatchNormalization(),
                ]
            )
        else:
            self.shortcut = None

    def call(self, inputs, training=None):
        """
        Forward pass in the residual block.

        Args:
            inputs (tensorflow.Tensor): Tensor of inputs to be passed to the block.
            training (bool): Whether or not to train batch normalization layers.

        Returns:
            tensorflow.Tensor: Block output tensor.
        """
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        shortcut = self.shortcut(inputs) if self.shortcut else inputs
        return self.relu(x + shortcut)


def light_upsample_block(x, skip_connection, filters, name_prefix):
    """
    Light upsample block using UpSampling2D and one convolutional layer.

    Args:
        x (tensorflow.Tensor): Tensor of inputs to be passed to the block.
        skip_connection (tensorflow.Tensor): Tensor to be used as skip connection.
        filters (int): Number of filters in the convolutional layers.
        name_prefix (str): Name prefix for the layers in the block.

    Returns:
        tensorflow.Tensor: Block output tensor.
    """
    # Bilinear upsampling followed by convolution
    x = tf.keras.layers.UpSampling2D(
        size=2, interpolation="bilinear", name=f"{name_prefix}_upsample"
    )(x)

    # Concatenate skip connection
    if skip_connection is not None:
        x = tf.keras.layers.Concatenate(name=f"{name_prefix}_concat")([x, skip_connection])

    # Two conv layers for better feature processing
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", name=f"{name_prefix}_conv1")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = tf.keras.layers.LeakyReLU(0.2, name=f'{name_prefix}_leaky_relu')(x)

    return x


def upsample_block(x, skip_connection, filters, name_prefix):
    """
    Upsample block using UpSampling2D and tw convolutional layers.

    Args:
        x (tensorflow.Tensor): Tensor of inputs to be passed to the block.
        skip_connection (tensorflow.Tensor): Tensor to be used as skip connection.
        filters (int): Number of filters in the convolutional layers.
        name_prefix (str): Name prefix for the layers in the block.

    Returns:
        tensorflow.Tensor: Block output tensor.
    """
    x = tf.keras.layers.Conv2DTranspose(filters, 2, 2, activation="relu")(x)

    # Concatenate skip connection
    if skip_connection is not None:
        x = tf.keras.layers.Concatenate(name=f"{name_prefix}_concat")([x, skip_connection])

    # Two conv layers for better feature processing
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", name=f"{name_prefix}_conv1")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = tf.keras.layers.LeakyReLU(0.2, name=f'{name_prefix}_leaky_relu1')(x)

    x = tf.keras.layers.Conv2D(filters, 3, padding="same", name=f"{name_prefix}_conv2")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = tf.keras.layers.LeakyReLU(0.2, name=f'{name_prefix}_leaky_relu2')(x)

    return x


def ResNet(input_shape, block_sizes, name="ResNet", mode="classification", num_classes=None):

    """
    Base ResNet model.

    Args:
        input_shape (list): Model input shape, in the format [h, w, c]
        block_sizes (list): List of block sizes.
        name (int): Model name. Defaulst to 'ResNet'.
        mode (str): Name of the task. Can be 'classification', 'segmentation', 'colorization', 'inpainting'.
        num_classes (int): Number of classes, if in classification mode.

    Returns:
        tensorflow.keras.Model: Created model for the given configuration
    """

    assert mode in ['classification', 'segmentation', 'colorization', 'inpainting'], \
           "Invalid mode. Choose either 'classification', 'segmentation', 'colorization' or 'inpainting'."
    
    num_classes = num_classes if num_classes else 20

    inputs = tf.keras.Input(shape=input_shape)

    # Initial layers
    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding="same", name="conv1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    # Store skip connections
    skip_connections = []

    # ResNet blocks
    filters = 64
    for i, size in enumerate(block_sizes):
        skip_connections.append(x)  # Store skip connection
        for j in range(size):
            stride = 2 if (i > 0 and j == 0) or (i == 0 and j == 0) else 1
            x = ResNetBlock(filters, stride=stride, name=f"block_{i}_{j}")(x)
        filters *= 2

    if mode == 'classification':
        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool_cls")(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='sigmoid', name="predictions_cls")(x)

    elif mode == 'segmentation':
        # Decoder pathway with skip connections
        skips = skip_connections[::-1]  # Reverse skip connections
        decoder_filters = [256, 128, 64, 32, 16]

        for i, filters in enumerate(decoder_filters):
            skip = skips[i] if i < len(skips) else None
            x = light_upsample_block(x, skip, filters, f"decoder_seg_{i}")

        # Final output layer
        outputs = tf.keras.layers.Conv2D(4, 3, padding="same", activation="softmax", name="decoder_output_conv_seg")(
            x
        )

    elif mode == 'colorization':
        # Decoder pathway with skip connections
        skips = skip_connections[::-1]  # Reverse skip connections
        decoder_filters = [256, 128, 64, 32, 16]

        for i, filters in enumerate(decoder_filters):
            skip = skips[i] if i < len(skips) else None
            x = light_upsample_block(x, skip, filters, f"decoder_{i}")

        # Final output layer
        outputs = tf.keras.layers.Conv2D(2, 3, padding="same", activation="sigmoid", name="decoder_output_conv")(
            x
        )

    elif mode == 'inpainting':
        # Decoder pathway with skip connections
        skips = skip_connections[::-1]  # Reverse skip connections
        decoder_filters = [256, 128, 64, 32, 16]

        for i, filters in enumerate(decoder_filters):
            skip = skips[i] if i < len(skips) else None
            x = upsample_block(x, skip, filters, f"decoder_{i}")

        # Final output layers
        x = tf.keras.layers.Conv2D(8, 3, padding="same", name="pre_output_conv")(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        outputs = tf.keras.layers.Conv2D(
            3, 3, padding="same", activation="sigmoid", name="output_conv"
        )(x)

    return tf.keras.Model(inputs, outputs, name=name)


def load_encoder_weights(model, weights_path):
    """
    Load weights by name, so only layers with same name and shape will be loaded.

    Args:
        model (tensorflow.keras.Model): Model to be used when loading weights.
        weights_path (str): Path to the saved weights to be loaded.
    """
    model.load_weights(weights_path, skip_mismatch=False, by_name=True)


def ResNet18(input_shape=(224, 224, 3), mode="classification", num_classes=None):
    """
    ResNet18 model.

    Args:
        input_shape (list): Model input shape, in the format [h, w, c].
        mode (str): Name of the task. Can be 'classification', 'segmentation', 'colorization', 'inpainting'.
        num_classes (int): Number of classes, if in classification mode.
    """
    return ResNet(input_shape, [2, 2, 2, 2], name="ResNet18", mode=mode, num_classes=num_classes)


def ResNet34(input_shape=(224, 224, 3), mode="classification", num_classes=None):
    """
    ResNet34 model.

    Args:
        input_shape (list): Model input shape, in the format [h, w, c].
        mode (str): Name of the task. Can be 'classification', 'segmentation', 'colorization', 'inpainting'.
        num_classes (int): Number of classes, if in classification mode.
    """
    return ResNet(input_shape, [3, 4, 6, 3], name="ResNet34", mode=mode, num_classes=num_classes)


def ResNet50(input_shape=(224, 224, 3), mode="classification", num_classes=None):
    """
    ResNet50 model.

    Args:
        input_shape (list): Model input shape, in the format [h, w, c].
        mode (str): Name of the task. Can be 'classification', 'segmentation', 'colorization', 'inpainting'.
        num_classes (int): Number of classes, if in classification mode.
    """
    return ResNet(input_shape, [3, 4, 6, 3], name="ResNet50", mode=mode, num_classes=num_classes)
