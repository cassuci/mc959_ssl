# src/models/base_model.py

import tensorflow as tf

class BaseModel(tf.keras.Model):
    def __init__(self, name="BaseModel", **kwargs):
        super().__init__(name=name, **kwargs)
        self.encoder = None
        self.decoder = None

    def build_encoder(self, input_shape):
        """Build the encoder part of the model."""
        raise NotImplementedError("Subclasses must implement build_encoder method.")

    def build_decoder(self, encoder_output_shape):
        """Build the decoder part of the model."""
        raise NotImplementedError("Subclasses must implement build_decoder method.")

    def call(self, inputs):
        """Forward pass of the model."""
        x = self.encoder(inputs)
        return self.decoder(x)

    def get_config(self):
        """Get the model configuration."""
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        """Create a model instance from configuration."""
        return cls(**config)

    def summary(self, **kwargs):
        """Print a string summary of the network."""
        inputs = tf.keras.Input(shape=self.input_shape[1:])
        model = tf.keras.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary(**kwargs)

    def plot_model(self, **kwargs):
        """Plot the model architecture."""
        inputs = tf.keras.Input(shape=self.input_shape[1:])
        model = tf.keras.Model(inputs=inputs, outputs=self.call(inputs))
        return tf.keras.utils.plot_model(model, **kwargs)

    def save_weights(self, filepath, **kwargs):
        """Save the model weights."""
        super().save_weights(filepath, **kwargs)

    def load_weights(self, filepath, **kwargs):
        """Load the model weights."""
        super().load_weights(filepath, **kwargs)

    def fine_tune(self, num_classes, trainable_layers=3):
        """Prepare the model for fine-tuning on a new task."""
        for layer in self.encoder.layers[:-trainable_layers]:
            layer.trainable = False
        
        x = self.encoder.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        self.fine_tuned_model = tf.keras.Model(inputs=self.encoder.input, outputs=outputs)
        return self.fine_tuned_model