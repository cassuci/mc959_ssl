import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.resnet import ResNet18

def get_file_paths(data_dir):
    """Get file paths for inpainting task."""
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
    
    return masked_files, original_files

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

def create_dataset(data_dir, batch_size):
    """Creates a TensorFlow dataset for inpainting."""
    # Get file paths
    masked_files, original_files = get_file_paths(data_dir)
    
    # Create datasets from file paths
    masked_dataset = tf.data.Dataset.from_tensor_slices(masked_files)
    original_dataset = tf.data.Dataset.from_tensor_slices(original_files)
    
    # Zip the datasets together
    dataset = tf.data.Dataset.zip((masked_dataset, original_dataset))
    
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
    
    return dataset, len(masked_files)  # Return dataset and number of samples

def train_inpainting(data_dir, model, epochs=10, batch_size=16):
    """Main training function for inpainting."""
    # Create dataset and get number of samples
    dataset, num_samples = create_dataset(data_dir, batch_size)
    
    # Calculate steps per epoch
    steps_per_epoch = num_samples // batch_size
    if steps_per_epoch == 0:
        # Adjust batch size if needed
        batch_size = min(num_samples, batch_size)
        steps_per_epoch = max(1, num_samples // batch_size)
        print(f"Adjusted batch size to {batch_size} for {num_samples} samples")
    
    print(f"Training with {num_samples} samples")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Batch size: {batch_size}")
    
    # Define loss and metrics
    loss = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.MeanSquaredError()]
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=loss,
        metrics=metrics
    )
    
    # Train with progress bar
    history = model.fit(
        dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=1
    )
    
    return history

if __name__ == "__main__":
    # Set up paths and print the full path for debugging
    data_dir = os.path.join("/mnt/f/ssl_images/data", "processed", "coco", "inpainting")
    print(f"Looking for data in: {data_dir}")
    
    # Initialize model
    model = ResNet18()
    model.build([16, 224, 224, 3])  # Build with RGB input shape
    
    print("Training inpainting model...")
    history = train_inpainting(data_dir, model)
    
    # Save the trained model
    os.makedirs("models", exist_ok=True)  # Create models directory if it doesn't exist
    save_path = os.path.join("models", "inpainting_model.h5")
    model.save_weights(save_path)
    print(f"Model saved to {save_path}")
    print("Inpainting training completed successfully!")