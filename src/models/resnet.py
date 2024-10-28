import tensorflow as tf

class ResNetBlock(tf.keras.layers.Layer):
    def __init__(self, filters, stride=1, name=None):
        super().__init__(name=name)
        self.filters = filters
        self.stride = stride
        
        # Define layers with unique names
        self.conv1 = tf.keras.layers.Conv2D(
            filters, 3, stride, padding="same", 
            name=f"{name}_conv1" if name else None
        )
        self.bn1 = tf.keras.layers.BatchNormalization(name=f"{name}_bn1" if name else None)
        self.conv2 = tf.keras.layers.Conv2D(
            filters, 3, padding="same",
            name=f"{name}_conv2" if name else None
        )
        self.bn2 = tf.keras.layers.BatchNormalization(name=f"{name}_bn2" if name else None)
        self.shortcut = None

    def build(self, input_shape):
        input_filters = input_shape[-1]
        if self.stride != 1 or input_filters != self.filters:
            self.shortcut = tf.keras.Sequential([
                tf.keras.layers.Conv2D(self.filters, 1, self.stride),
                tf.keras.layers.BatchNormalization(),
            ])
        super().build(input_shape)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.shortcut:
            shortcut = self.shortcut(inputs)
        else:
            shortcut = inputs
            
        return tf.nn.relu(x + shortcut)

class ResNet(tf.keras.Model):
    def __init__(self, block_sizes, name="ResNet"):
        super().__init__(name=name)
        self.block_sizes = block_sizes
        
        # Initial layers
        self.conv1 = tf.keras.layers.Conv2D(64, 7, strides=2, padding="same", name="conv1")
        self.bn1 = tf.keras.layers.BatchNormalization(name="bn1")
        self.pool1 = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same", name="pool1")
        
        # ResNet blocks
        self.blocks = []
        filters = 64
        for i, size in enumerate(block_sizes):
            block_list = []
            for j in range(size):
                stride = 2 if i > 0 and j == 0 else 1
                block_list.append(
                    ResNetBlock(filters, stride=stride, name=f"block_{i}_{j}")
                )
            self.blocks.append(block_list)
            filters *= 2
            
        # Decoder layers
        self.decoder_blocks = [
            # Each decoder block: (filters, strides)
            (256, 2), (128, 2), (64, 2), (32, 2), (16, 2)
        ]
        
        self.decoder_layers = []
        for i, (filters, strides) in enumerate(self.decoder_blocks):
            self.decoder_layers.extend([
                tf.keras.layers.Conv2DTranspose(
                    filters, 4, strides=strides, padding="same",
                    name=f"deconv{i+1}"
                ),
                tf.keras.layers.BatchNormalization(name=f"debn{i+1}"),
                tf.keras.layers.ReLU(name=f"derelu{i+1}")
            ])
            
        # Output layer
        self.output_conv = tf.keras.layers.Conv2D(
            3, 3, padding="same", activation="sigmoid", name="output_conv"
        )

    def encode(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        
        # ResNet blocks
        for block_list in self.blocks:
            for block in block_list:
                x = block(x)
                
        return x
    
    def decode(self, x):
        # Decoder blocks
        for layer in self.decoder_layers:
            x = layer(x)
            
        # Output
        return self.output_conv(x)

    def call(self, inputs):
        x = self.encode(inputs)
        return self.decode(x)
    
    def save_weights(self, filepath, **kwargs):
        """Save model weights with proper format handling."""
        try:
            super().save_weights(filepath, save_format='h5', **kwargs)
        except Exception as e:
            print(f"H5 save failed ({str(e)}), trying TensorFlow format...")
            super().save_weights(filepath, save_format='tf', **kwargs)
            
    def load_weights(self, filepath, **kwargs):
        """Load model weights with proper format handling."""
        try:
            super().load_weights(filepath, **kwargs)
        except Exception as e:
            print(f"Direct load failed ({str(e)}), trying TensorFlow format...")
            super().load_weights(filepath + '.index', **kwargs)

def ResNet18():
    return ResNet([2, 2, 2, 2], name="ResNet18")

def ResNet34():
    return ResNet([3, 4, 6, 3], name="ResNet34")

def ResNet50():
    return ResNet([3, 4, 6, 3], name="ResNet50")