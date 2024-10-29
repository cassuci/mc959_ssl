import tensorflow as tf


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

        # Shortcut connection
        if stride != 1 or filters != 64:  # Adjust this for your architecture
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


def ResNet(input_shape, block_sizes, name="ResNet"):
    inputs = tf.keras.Input(shape=input_shape)

    # Initial layers - removed MaxPooling and adjusted stride
    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding="same", name="conv1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.ReLU()(x)

    # ResNet blocks - adjusted initial stride
    filters = 64
    for i, size in enumerate(block_sizes):
        for j in range(size):
            # Modified stride calculation since we removed MaxPooling
            stride = 2 if (i > 0 and j == 0) or (i == 0 and j == 0) else 1
            x = ResNetBlock(filters, stride=stride, name=f"block_{i}_{j}")(x)
        filters *= 2

    # Decoder layers - adjusted to account for removed MaxPooling
    x = tf.keras.layers.Conv2DTranspose(256, 4, strides=2, padding="same", name="deconv1")(x)
    x = tf.keras.layers.BatchNormalization(name="debn1")(x)
    x = tf.keras.layers.ReLU(name="derelu1")(x)
    x = tf.keras.layers.Conv2DTranspose(128, 4, strides=2, padding="same", name="deconv2")(x)
    x = tf.keras.layers.BatchNormalization(name="debn2")(x)
    x = tf.keras.layers.ReLU(name="derelu2")(x)
    x = tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding="same", name="deconv3")(x)
    x = tf.keras.layers.BatchNormalization(name="debn3")(x)
    x = tf.keras.layers.ReLU(name="derelu3")(x)
    x = tf.keras.layers.Conv2DTranspose(32, 4, strides=2, padding="same", name="deconv4")(x)
    x = tf.keras.layers.BatchNormalization(name="debn4")(x)
    x = tf.keras.layers.ReLU(name="derelu4")(x)
    x = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding="same", name="deconv5")(x)
    x = tf.keras.layers.BatchNormalization(name="debn5")(x)
    x = tf.keras.layers.ReLU(name="derelu5")(x)

    # Output layer
    outputs = tf.keras.layers.Conv2D(
        3, 3, padding="same", activation="sigmoid", name="output_conv"
    )(x)

    return tf.keras.Model(inputs, outputs, name=name)


def ResNet18(input_shape=(224, 224, 3)):
    return ResNet(input_shape, [2, 2, 2, 2], name="ResNet18")


def ResNet34(input_shape=(224, 224, 3)):
    return ResNet(input_shape, [3, 4, 6, 3], name="ResNet34")


def ResNet50(input_shape=(224, 224, 3)):
    return ResNet(input_shape, [3, 4, 6, 3], name="ResNet50")
