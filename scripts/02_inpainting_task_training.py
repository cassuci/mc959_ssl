import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.resnet import ResNet18

def get_file_paths(data_dir):
    """Get file paths for inpainting task and split into train/val sets."""
    # Ensure the directory exists
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
    
    files = os.listdir(data_dir)
    
    masked_files = sorted([
        os.path.join(data_dir, f) for f in files 
        if f.startswith("masked")
    ])
    
    original_files = sorted([
        os.path.join(data_dir, f) for f in files 
        if f.startswith("original")
    ])
    
    # Print debugging information
    print(f"Found {len(masked_files)} masked files and {len(original_files)} original files")
    
    if len(masked_files) == 0 or len(original_files) == 0:
        raise ValueError(f"No files found in {data_dir} with required prefixes (masked/original)")
    
    if len(masked_files) != len(original_files):
        raise ValueError("Number of masked and original files don't match")
    
    # Calculate split indices (80% train, 20% validation)
    total_samples = len(masked_files)
    train_size = int(0.8 * total_samples)
    
    # Split the files
    train_masked = masked_files[:train_size]
    train_original = original_files[:train_size]
    val_masked = masked_files[train_size:]
    val_original = original_files[train_size:]
    
    print(f"Split dataset into {len(train_masked)} training and {len(val_masked)} validation samples")
    
    return (train_masked, train_original), (val_masked, val_original)

def load_image(file_path):
    """Loads and preprocesses an image for inpainting."""
    file_path = file_path.numpy().decode("utf-8")
    
    try:
        img = np.load(file_path)
        img = tf.cast(img, tf.float32) / 255.0
        return img
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        raise

def create_dataset(masked_files, original_files, batch_size, shuffle=True):
    """Creates a TensorFlow dataset for inpainting."""
    # Create datasets from file paths
    masked_dataset = tf.data.Dataset.from_tensor_slices(masked_files)
    original_dataset = tf.data.Dataset.from_tensor_slices(original_files)
    
    # Zip the datasets together
    dataset = tf.data.Dataset.zip((masked_dataset, original_dataset))
    
    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    def load_image_pair(masked_path, original_path):
        masked_img = tf.py_function(load_image, [masked_path], tf.float32)
        original_img = tf.py_function(load_image, [original_path], tf.float32)
        
        # Set shapes explicitly
        masked_img.set_shape([224, 224, 3])
        original_img.set_shape([224, 224, 3])
        
        return masked_img, original_img
    
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

def train_inpainting(data_dir, model, epochs=10, batch_size=16):
    """Main training function for inpainting with validation."""
    # Print model summary
    print("\nModel Architecture:")
    print("=" * 50)
    model.summary()
    print("=" * 50, "\n")
    
    # Get train and validation file paths
    (train_masked, train_original), (val_masked, val_original) = get_file_paths(data_dir)
    
    # Create training and validation datasets
    train_dataset = create_dataset(train_masked, train_original, batch_size, shuffle=True)
    val_dataset = create_dataset(val_masked, val_original, batch_size, shuffle=False)
    
    # Calculate steps per epoch
    train_steps = len(train_masked) // batch_size
    val_steps = len(val_masked) // batch_size
    
    if train_steps == 0:
        # Adjust batch size if needed
        batch_size = min(len(train_masked), batch_size)
        train_steps = max(1, len(train_masked) // batch_size)
        print(f"Adjusted batch size to {batch_size} for {len(train_masked)} training samples")
    
    # Print training configuration
    print("Training Configuration:")
    print(f"Total training samples: {len(train_masked)}")
    print(f"Total validation samples: {len(val_masked)}")
    print(f"Training steps per epoch: {train_steps}")
    print(f"Validation steps per epoch: {val_steps}")
    print(f"Batch size: {batch_size}")
    print("=" * 50, "\n")
    
    # Define loss and metrics
    loss = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.MeanSquaredError()]
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=loss,
        metrics=metrics
    )
    
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
        PeriodicCheckpoint(save_freq=5, checkpoint_dir=checkpoint_dir),
        # Reduce learning rate when plateauing
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
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
    # Set up paths and print the full path for debugging
    data_dir = os.path.join("/mnt/f/ssl_images/data", "processed", "coco", "inpainting")
    print(f"Looking for data in: {data_dir}")
    
    # Initialize model
    model = ResNet18((224, 224, 3))
    
    print("Training inpainting model...")
    history = train_inpainting(data_dir, model, epochs=30)
    
    # Save the final model
    os.makedirs("models", exist_ok=True)  # Create models directory if it doesn't exist
    save_path = os.path.join("models", "inpainting_model_final.h5")
    model.save_weights(save_path)
    print(f"Final model saved to {save_path}")
    print("Inpainting training completed successfully!")