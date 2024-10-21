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
    def __init__(self, block_sizes, name="ResNet", min_input_shape=None, include_top=True, initial_weights=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.block_sizes = block_sizes
        self.min_input_shape = min_input_shape
        self.include_top = include_top
        self.initial_weights = initial_weights
        self.encoder = None
        self.decoder = None

    def build(self, input_shape):
        if self.min_input_shape is not None and input_shape[1:] < self.min_input_shape:
            input_shape = (input_shape[0], *self.min_input_shape)
        inputs = tf.keras.Input(shape=input_shape[1:])
        
        # Initial Conv Layer
        x = tf.keras.layers.Conv2D(64, 7, strides=2, padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Store intermediate outputs for skip connections
        skips = []
        filters = 64
        
        # ResNet Blocks
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
        
        # Start upsampling from 7x7x512
        x = main_output  # Starting with the output of the last block

        # Loop over the skip connections in reverse order
        for i, skip in enumerate(reversed(skip_outputs)):
            x = tf.keras.layers.Conv2DTranspose(x.shape[-1] // 2, 3, strides=2, padding="same")(x)  # Upsample
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

            # Resize skip connection to match the upsampled feature map
            skip_resized = tf.keras.layers.Resizing(height=x.shape[1], width=x.shape[2])(skip)
            x = tf.keras.layers.Concatenate()([x, skip_resized])

            # Ensure ResNetBlock has correct number of filters
            x = ResNetBlock(x.shape[-1])(x)

        # Adjusting the output size in the decoder
        x = tf.keras.layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')(x)  # -> 448x448
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        # Resize to match the expected dimensions (if needed)
        x = tf.keras.layers.Resizing(height=224, width=224)(x)  # Now resize to 224x224

        x = tf.keras.layers.Conv2D(3, kernel_size=1, activation='sigmoid')(x)  # Final output layer

        outputs = tf.keras.layers.Conv2D(3, kernel_size=3, padding="same", activation="sigmoid")(x)  # Final output: 224x224x3

        # Create the decoder model using the same `inputs`
        self.decoder = tf.keras.Model(inputs=inputs, outputs=outputs, name="decoder")

        # Print the model summary
        self.encoder.summary()
        self.decoder.summary()

        super().build(input_shape)

    def call(self, inputs):
        if self.min_input_shape is not None and inputs.shape[1:] < self.min_input_shape:
            inputs = tf.keras.layers.Resizing(*self.min_input_shape)(inputs)
        return self.decoder(inputs)

def ResNet18():
    return ResNet([2, 2, 2, 2], name="ResNet18")

def ResNet34():
    return ResNet([3, 4, 6, 3], name="ResNet34")

def ResNet50():
    return ResNet([3, 4, 6, 3], name="ResNet50")
