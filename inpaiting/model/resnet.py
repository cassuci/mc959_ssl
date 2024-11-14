import tensorflow as tf
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from weights.load_weights import load_model_weights
from weights.weights import weights_collection


class ResNetBlock(tf.keras.Model):
    def __init__(self, filters, stride=1, name=None):
        super().__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2D(
            filters, 3, strides=stride, padding="same", name=f"{name}_conv1"
        )
        self.bn1 = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")
        self.relu = tf.keras.layers.ReLU()
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
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        shortcut = self.shortcut(inputs) if self.shortcut else inputs
        return self.relu(x + shortcut)


def upsample_block(x, skip_connection, filters, name_prefix):
    x = tf.keras.layers.Conv2DTranspose(filters, 2, 2, activation="relu")(x)

    # Concatenate skip connection
    if skip_connection is not None:
        x = tf.keras.layers.Concatenate(name=f"{name_prefix}_concat")([x, skip_connection])

    # Two conv layers for better feature processing
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", name=f"{name_prefix}_conv1")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters, 3, padding="same", name=f"{name_prefix}_conv2")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = tf.keras.layers.ReLU()(x)

    return x


def ResNet(input_shape, block_sizes, name="ResNet", mode="inpainting"):
    assert mode in [
        "classification",
        "inpainting",
    ], "Invalid mode. Choose either 'classification', 'colorization' or 'inpainting'."

    inputs = tf.keras.Input(shape=input_shape)

    # Initial layers
    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding="same", name="conv1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.ReLU()(x)

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

    if mode == "classification":
        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
        # x = tf.keras.layers.Flatten(name='flatten')(x)
        # x = tf.keras.layers.Dense(500, activation='relu', name='cls_1')(x)
        # x = tf.keras.layers.Dense(500, activation='relu', name='cls_2')(x)
        outputs = tf.keras.layers.Dense(20, activation="sigmoid", name="predictions")(x)

    elif mode == "inpainting":
        # Decoder pathway with skip connections
        skips = skip_connections[::-1]  # Reverse skip connections
        decoder_filters = [256, 128, 64, 32, 16]

        for i, filters in enumerate(decoder_filters):
            skip = skips[i] if i < len(skips) else None
            x = upsample_block(x, skip, filters, f"decoder_{i}")

        # Final output layers
        x = tf.keras.layers.Conv2D(8, 3, padding="same", name="pre_output_conv")(x)
        x = tf.keras.layers.ReLU()(x)
        outputs = tf.keras.layers.Conv2D(
            3, 3, padding="same", activation="sigmoid", name="output_conv"
        )(x)

    return tf.keras.Model(inputs, outputs, name=name)


def load_encoder_weights(model, weights_path):
    """Load weights by name, so only layers with same name and shape will be loaded."""
    model.load_weights(weights_path)


def ResNet18(input_shape=(224, 224, 3), mode="classification"):
    model = ResNet(input_shape, [2, 2, 2, 2], name="ResNet18", mode=mode)
    model = load_model_weights(weights_collection, model, 1000, False, "resnet18")
    return model
