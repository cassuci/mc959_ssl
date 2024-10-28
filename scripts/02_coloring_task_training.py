import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.resnet import ResNet18, ResNet50

def get_file_paths(data_dir):
    """Get file paths for colorization task and split into train/val sets."""
    files = os.listdir(data_dir)
    
    gray_files = sorted([
        os.path.join(data_dir, f) for f in files
        if f.startswith("gray")
    ])
    
    color_files = sorted([
        os.path.join(data_dir, f) for f in files
        if f.startswith("color")
    ])
    
    # Calculate split indices (80% train, 20% validation)
    total_samples = len(gray_files)
    train_size = int(0.8 * total_samples)
    
    # Split the files
    train_gray = gray_files[:train_size]
    train_color = color_files[:train_size]
    val_gray = gray_files[train_size:]
    val_color = color_files[train_size:]
    
    return (train_gray, train_color), (val_gray, val_color)

def load_image(file_path):
    """Loads and preprocesses an image for colorization."""
    # Convert the EagerTensor back to a numpy string
    file_path = file_path.numpy().decode("utf-8")
    
    # Load the .npy file content
    img = np.load(file_path)
    
    # Preprocess the image
    img = tf.cast(img, tf.float32) / 255.0
    
    if "gray" in file_path:
        # Ensure grayscale input is 1 channel
        if len(img.shape) == 3 and img.shape[-1] == 3:
            img = tf.image.rgb_to_grayscale(img)
    
    return img

def create_dataset(gray_files, color_files, batch_size, shuffle=True):
    """Creates a TensorFlow dataset for colorization."""
    # Create datasets from file paths
    gray_dataset = tf.data.Dataset.from_tensor_slices(gray_files)
    color_dataset = tf.data.Dataset.from_tensor_slices(color_files)
    
    # Zip the datasets together
    dataset = tf.data.Dataset.zip((gray_dataset, color_dataset))
    
    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    # Function to load both images
    def load_image_pair(gray_path, color_path):
        gray_img = tf.py_function(load_image, [gray_path], tf.float32)
        color_img = tf.py_function(load_image, [color_path], tf.float32)
        
        # Set shapes explicitly
        gray_img.set_shape([224, 224, 1])
        color_img.set_shape([224, 224, 3])
        
        return gray_img, color_img
    
    # Map the loading function
    dataset = dataset.map(load_image_pair, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

class PeriodicCheckpoint(tf.keras.callbacks.Callback):
    """Custom callback to save model weights periodically."""
    def __init__(self, save_freq=5, checkpoint_dir='checkpoints'):
        super().__init__()
        self.save_freq = save_freq
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, 
                f'model_epoch_{epoch + 1:03d}.h5'
            )
            self.model.save_weights(checkpoint_path)
            print(f"\nSaved periodic checkpoint for epoch {epoch + 1} at: {checkpoint_path}")

def train_colorization(data_dir, model, epochs=10, batch_size=16):
    """Main training function for colorization with validation."""
    # Print model summary
    print("\nModel Architecture:")
    print("=" * 50)
    model.summary()
    print("=" * 50, "\n")
    
    # Get train and validation file paths
    (train_gray, train_color), (val_gray, val_color) = get_file_paths(data_dir)
    
    # Create training and validation datasets
    train_dataset = create_dataset(train_gray, train_color, batch_size, shuffle=True)
    val_dataset = create_dataset(val_gray, val_color, batch_size, shuffle=False)
    
    # Define loss and metrics
    loss = tf.keras.losses.MeanAbsoluteError()
    metrics = [tf.keras.metrics.MeanAbsoluteError()]
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=loss,
        metrics=metrics
    )
    
    # Calculate steps per epoch
    train_steps = len(train_gray) // batch_size
    val_steps = len(val_gray) // batch_size
    
    # Print training configuration
    print("Training Configuration:")
    print(f"Total training samples: {len(train_gray)}")
    print(f"Total validation samples: {len(val_gray)}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {train_steps}")
    print(f"Validation steps: {val_steps}")
    print("=" * 50, "\n")
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join("models", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Add callbacks
    callbacks = [
        # Early stopping callback
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        # Best model checkpoint callback
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True
        ),
        # Periodic checkpoint callback (every 5 epochs)
        PeriodicCheckpoint(save_freq=5, checkpoint_dir=checkpoint_dir)
    ]
    
    # Train with progress bar and validation
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

if __name__ == "__main__":
    # Set up paths
    data_dir = os.path.join("/mnt/f/ssl_images/data", "processed", "coco", 'colorization')
    
    # Initialize model (assuming your ResNet18 is modified for colorization)
    model = ResNet50((224, 224, 1))
    
    print("Training colorization model...")
    history = train_colorization(data_dir, model, epochs=100)
    
    # Save the final model
    save_path = os.path.join("models", "colorization_model_resnet50.h5")
    model.save_weights(save_path)
    print(f"Final model saved to {save_path}")
    print("Colorization training completed successfully!")