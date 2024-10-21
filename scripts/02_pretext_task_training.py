import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm  # optional for progress bar

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.resnet import ResNet18


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


def load_image(file_path, task):
    """Loads an image from an `.npy` file and preprocesses it."""
    # Convert the EagerTensor back to a numpy string
    file_path = file_path.numpy().decode("utf-8")

    # Load the .npy file content
    img = np.load(file_path)  # Load image data

    # Preprocess the image (as needed)
    img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0, 1]

    if task == "inpainting":
        # Implement any specific inpainting logic here
        return img, img  # Return masked and original (example logic)

    elif task == "colorization":
        # Implement any specific colorization logic here
        return img, img  # Return gray and color images (example logic)

    else:
        raise ValueError(f"Unsupported task: {task}")


def create_dataset(data_dir, task, batch_size):
    """Creates a TensorFlow dataset from `.npy` files."""

    # Get the file paths for the specified task
    task_dir = os.path.join(data_dir, task)

    # Identify the maximum suffix number to construct file paths
    max_suffix = 0
    for f in os.listdir(task_dir):
        if f.endswith(".npy") and (f.startswith("masked_") or f.startswith("gray_")):
            suffix = int(f.split("_")[-1].split(".")[0])  # Extract the numerical suffix
            max_suffix = max(max_suffix, suffix)

    # Generate the file paths based on the maximum suffix
    file_paths = []
    for i in range(max_suffix + 1):
        if task == "inpainting":
            masked_file = os.path.join(task_dir, f"masked_{i}.npy")
            if os.path.exists(masked_file):
                file_paths.append(masked_file)
        elif task == "colorization":
            gray_file = os.path.join(task_dir, f"gray_{i}.npy")
            if os.path.exists(gray_file):
                file_paths.append(gray_file)

    # Create a TensorFlow dataset from the file paths
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)

    # Function to wrap load_image for use with tf.py_function
    def load_image_wrapper(file_path):
        return tf.py_function(
            func=load_image, inp=[file_path, task], Tout=[tf.float32, tf.float32]
        )

    # Map the load_image_wrapper to the dataset
    dataset = dataset.map(load_image_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch the dataset
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


def get_loss_and_metrics(task):
    # Define loss and metrics based on task
    if task == "inpainting":
        loss = tf.keras.losses.MeanSquaredError()
        metrics = [tf.keras.metrics.MeanSquaredError()]
    elif task == "colorization":
        loss = tf.keras.losses.MeanAbsoluteError()
        metrics = [tf.keras.metrics.MeanAbsoluteError()]
    else:
        raise ValueError(f"Unsupported task: {task}")
    return loss, metrics


def train_pretext_task(task, data_dir, model, epochs=10, batch_size=16):
    dataset = create_dataset(data_dir, task, batch_size)
    loss, metrics = get_loss_and_metrics(task)

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics=metrics)
    steps_per_epoch = len(get_file_paths(data_dir, task)[0]) // batch_size

    # Train with progress bar (optional)
    history = model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1)
    # Alternatively, use verbose=2 for more detailed output

    return history


if __name__ == "__main__":
    data_dir = os.path.join("/mnt/f/ssl_images/data", "processed", "coco")
    model = ResNet18()
    model.build([16, 224, 224, 3])  # Build the model

    # Train on inpainting and colorization tasks (separated functions)
    print("Training on inpainting task...")
    inpainting_history = train_pretext_task("inpainting", data_dir, model)

    print("Training on colorization task...")
    colorization_history = train_pretext_task("colorization", data_dir, model)

    # Save the trained model
    model.save_weights(os.path.join("models", "pretrained_resnet18.h5"))
    print("Pretext task training completed successfully!")
