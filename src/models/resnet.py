# src/models/resnet.py

import tensorflow as tf
from .base_model import BaseModel

class ResNetBlock(tf.keras.layers.Layer):
    def __init__(self, filters, stride=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, 3, stride, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        
        if stride != 1:
            self.shortcut = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters, 1, stride),
                tf.keras.layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x

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

    def build_encoder(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape[1:])
        x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)

        filters = 64
        for size in self.block_sizes:
            for _ in range(size):
                x = ResNetBlock(filters)(x)
            filters *= 2

        self.encoder = tf.keras.Model(inputs=inputs, outputs=x, name="encoder")

    def build_decoder(self, encoder_output_shape):
        inputs = tf.keras.Input(shape=encoder_output_shape[1:])
        x = inputs

        for _ in range(2):  # Adjust the number of upsampling layers as needed
            x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        outputs = tf.keras.layers.Conv2D(3, 3, padding='same', activation='sigmoid')(x)
        self.decoder = tf.keras.Model(inputs=inputs, outputs=outputs, name="decoder")

    def call(self, inputs):
        x = self.encoder(inputs)
        return self.decoder(x)

def ResNet18():
    return ResNet([2, 2, 2, 2], name="ResNet18")

def ResNet34():
    return ResNet([3, 4, 6, 3], name="ResNet34")

def ResNet50():
    return ResNet([3, 4, 6, 3], name="ResNet50")