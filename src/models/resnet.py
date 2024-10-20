import tensorflow as tf
from .base_model import BaseModel


class ResNetBlock(tf.keras.layers.Layer):
    def __init__(self, filters, stride=1):
        super().__init__()
        self.filters = filters
        self.stride = stride
        self.conv1 = tf.keras.layers.Conv2D(filters, 3, stride, padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.shortcut = None

    def build(self, input_shape):
        input_filters = input_shape[-1]
        if self.stride != 1 or input_filters != self.filters:
            self.shortcut = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(self.filters, 1, self.stride),
                    tf.keras.layers.BatchNormalization(),
                ]
            )
        else:
            self.shortcut = lambda x: x
        super().build(input_shape)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        shortcut = self.shortcut(inputs)
        return tf.nn.relu(x + shortcut)


class ResNet(BaseModel):
    def __init__(self, block_sizes, name="ResNet", **kwargs):
        super().__init__(name=name, **kwargs)
        self.block_sizes = block_sizes
        self.encoder = None
        self.decoder = None

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape[1:])
        x = tf.keras.layers.Conv2D(64, 7, strides=2, padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Store intermediate outputs for skip connections
        skips = []

        filters = 64
        for i, size in enumerate(self.block_sizes):
            for j in range(size):
                stride = 2 if i > 0 and j == 0 else 1
                x = ResNetBlock(filters, stride=stride)(x)
            skips.append(x)
            filters *= 2

        # Encoder model
        self.encoder = tf.keras.Model(inputs=inputs, outputs=[x] + skips, name="encoder")

        # Get the encoder outputs
        encoder_outputs = self.encoder(inputs)
        main_output, *skip_outputs = encoder_outputs
        x = main_output

        # Decoder logic: start upsampling from 7x7x512 back to 224x224x3
        for i, skip in enumerate(reversed(skip_outputs)):
            x = tf.keras.layers.Conv2DTranspose(x.shape[-1] // 2, 3, strides=2, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

            # Resize skip connection to match the upsampled feature map
            skip_resized = tf.keras.layers.Resizing(x.shape[1], x.shape[2])(skip)
            x = tf.keras.layers.Concatenate()([x, skip_resized])

            x = ResNetBlock(x.shape[-1] // 2)(x)

        # Continue upsampling to reach 224x224 resolution
        x = tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding="same")(x)  # 14x14x256
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)  # 28x28x128
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)  # 56x56x64
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same")(x)  # 112x112x32
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2DTranspose(16, 3, strides=2, padding="same")(x)  # 224x224x16
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        outputs = tf.keras.layers.Conv2D(3, 3, padding="same", activation="sigmoid")(
            x
        )  # 224x224x3

        # Create the decoder model using the same `inputs`
        self.decoder = tf.keras.Model(inputs=inputs, outputs=outputs, name="decoder")
        super().build(input_shape)

    def call(self, inputs):
        return self.decoder(inputs)


def ResNet18():
    return ResNet([2, 2, 2, 2], name="ResNet18")


def ResNet34():
    return ResNet([3, 4, 6, 3], name="ResNet34")


def ResNet50():
    return ResNet([3, 4, 6, 3], name="ResNet50")
