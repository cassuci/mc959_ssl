import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.resnet import ResNet18  # Import your ResNet implementation

def get_file_paths(data_dir):
    """Get file paths for colorization task."""
    files = os.listdir(data_dir)
    
    gray_files = sorted([
        os.path.join(data_dir, f) for f in files 
        if f.startswith("gray")
    ])
    
    color_files = sorted([
        os.path.join(data_dir, f) for f in files 
        if f.startswith("color")
    ])
    
    return gray_files, color_files

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

def create_dataset(data_dir, batch_size):
    """Creates a TensorFlow dataset for colorization."""
    # Get file paths
    gray_files, color_files = get_file_paths(data_dir)
    
    # Create datasets from file paths
    gray_dataset = tf.data.Dataset.from_tensor_slices(gray_files)
    color_dataset = tf.data.Dataset.from_tensor_slices(color_files)
    
    # Zip the datasets together
    dataset = tf.data.Dataset.zip((gray_dataset, color_dataset))
    
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

def train_colorization(data_dir, model, epochs=10, batch_size=16):
    """Main training function for colorization."""
    # Create dataset
    dataset = create_dataset(data_dir, batch_size)
    
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
    gray_files, _ = get_file_paths(data_dir)
    steps_per_epoch = len(gray_files) // batch_size
    
    # Train with progress bar
    history = model.fit(
        dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=1
    )
    
    return history

if __name__ == "__main__":
    # Set up paths
    data_dir = os.path.join("/mnt/f/ssl_images/data", "processed", "coco", 'colorization')
    
    # Initialize model (assuming your ResNet18 is modified for colorization)
    model = ResNet18()
    model.build([None, 224, 224, 1])  # Build with grayscale input shape
    
    print("Training colorization model...")
    history = train_colorization(data_dir, model, epochs=1)
    
    # Save the trained model
    save_path = os.path.join("models", "colorization_model.h5")
    model.save_weights(save_path)
    print(f"Model saved to {save_path}")
    print("Colorization training completed successfully!")