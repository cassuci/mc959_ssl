import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.resnet import ResNet18
from src.utils.metrics import SegmentationMetrics


def get_file_paths(data_dir, task):
    """Get file paths for the specified task."""
    task_dir = os.path.join(data_dir, task)
    files = os.listdir(task_dir)

    if task == "inpainting":
        masked_files = sorted([os.path.join(task_dir, f) for f in files if f.startswith("masked")])
        original_files = sorted(
            [os.path.join(task_dir, f) for f in files if f.startswith("original")]
        )
        return masked_files, original_files
    elif task == "colorization":
        gray_files = sorted([os.path.join(task_dir, f) for f in files if f.startswith("gray")])
        color_files = sorted([os.path.join(task_dir, f) for f in files if f.startswith("color")])
        return gray_files, color_files


def load_image(file_path):
    """Load and preprocess a single image."""

    def load_npy(path):
        img = np.load(path.decode("utf-8"))  # Decode the path from bytes to string
        return img

    # Load image using tf.py_function
    img = tf.py_function(load_npy, [file_path], tf.float32)
    img.set_shape([None, None, None])  # Set shape to allow for flexible dimensions

    # Ensure img is a tensor and check dimensions
    img = tf.convert_to_tensor(img)

    # Print the image shape for debugging
    tf.print(f"Loaded image shape: {img.shape}")

    if img.shape.ndims not in (3, 4):  # Check for 3 or 4 dimensions
        raise ValueError(
            f"Image at {file_path} has unexpected number of dimensions: {img.shape.ndims}"
        )

    if img.shape.ndims == 3 and img.shape[-1] == 1:  # Check for grayscale (3D with single channel)
        img = tf.expand_dims(img, axis=-1)  # Add channel dimension if grayscale

    return tf.image.resize(img, (224, 224))


def create_dataset(input_paths, target_paths, batch_size):
    """Create a TensorFlow dataset from file paths."""
    input_ds = tf.data.Dataset.from_tensor_slices(input_paths)
    target_ds = tf.data.Dataset.from_tensor_slices(target_paths)

    input_ds = input_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    target_ds = target_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = tf.data.Dataset.zip((input_ds, target_ds))
    return dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_pretext_task(task, data_dir, model, epochs=10, batch_size=32):
    """Train the model on the specified pretext task."""
    input_paths, target_paths = get_file_paths(data_dir, task)
    dataset = create_dataset(input_paths, target_paths, batch_size)

    if task == "inpainting":
        loss = tf.keras.losses.MeanSquaredError()
        metrics = [tf.keras.metrics.MeanSquaredError()]
    elif task == "colorization":
        loss = tf.keras.losses.MeanAbsoluteError()
        metrics = [tf.keras.metrics.MeanAbsoluteError()]

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics=metrics)

    steps_per_epoch = len(input_paths) // batch_size
    history = model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1)
    return history


if __name__ == "__main__":
    data_dir = os.path.join("/mnt/f/ssl_images/data", "processed", "coco")
    model = ResNet18()

    # Build the model with a sample input
    sample_input = tf.keras.Input(shape=(224, 224, 3))
    model.build(sample_input.shape)

    # Train on inpainting task
    print("Training on inpainting task...")
    inpainting_history = train_pretext_task("inpainting", data_dir, model)

    # Train on colorization task
    print("Training on colorization task...")
    colorization_history = train_pretext_task("colorization", data_dir, model)

    # Save the trained model
    model.save_weights(os.path.join("models", "pretrained_resnet18.h5"))
    print("Pretext task training completed successfully!")
