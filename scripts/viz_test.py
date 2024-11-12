import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os
import tensorflow as tf
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.resnet import ResNet18


def load_and_preprocess_image(image_path):
    """Load and preprocess a single image for colorization."""
    # Load image
    img = np.load(image_path)

    # Convert to float and normalize
    img = tf.cast(img, tf.float32) / 255.0

    # Ensure grayscale input
    if len(img.shape) == 3 and img.shape[-1] == 3:
        img = tf.image.rgb_to_grayscale(img)

    # Add batch dimension
    img = tf.expand_dims(img, 0)

    return img


def visualize_colorization(model_path, image_path):
    """Load model, process image, and visualize results."""
    # Initialize model
    model = ResNet18((224, 224, 1))

    # Load trained weights
    model.load_weights(model_path)

    # Load and preprocess input image
    input_image = load_and_preprocess_image(image_path)

    # Get model prediction
    predicted_color = model.predict(input_image)

    # Remove batch dimension
    input_image = tf.squeeze(input_image)
    predicted_color = tf.squeeze(predicted_color)

    # Load original color image for comparison
    color_path = image_path.replace("gray", "color")
    original_color = np.load(color_path)
    original_color = original_color.astype(np.float32)  # / 255.0

    # Create figure with three subplots
    plt.figure(figsize=(15, 5))

    # Plot grayscale input
    plt.subplot(1, 3, 1)
    plt.imshow(input_image, cmap="gray")
    plt.title("Grayscale Input")
    plt.axis("off")

    # Plot model prediction
    plt.subplot(1, 3, 2)
    plt.imshow(predicted_color)
    plt.title("Model Colorization")
    plt.axis("off")

    # Plot original color image
    plt.subplot(1, 3, 3)
    plt.imshow(original_color)
    plt.title("Original Color")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Print some statistics
    mae = tf.reduce_mean(tf.abs(original_color - predicted_color))
    print(f"Mean Absolute Error: {mae:.4f}")


if __name__ == "__main__":
    # Set up paths
    model_path = os.path.join("models", "colorization_model_final.h5")
    data_dir = os.path.join("/mnt/f/ssl_images/data", "processed", "coco", "colorization")

    # Get a test image path (using the first grayscale image in the directory)
    test_image = next(
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith("gray")
    )

    print("Running colorization inference...")
    visualize_colorization(model_path, test_image)
    print("Visualization complete!")
