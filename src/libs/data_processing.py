import numpy as np
from PIL import Image
import tensorflow as tf
import json
import pycocotools.mask as mask_util


def load_image(image_path):
    """Load an image from a file path."""
    return Image.open(image_path)


def resize_image(image, size=(224, 224)):
    """Resize an image to a given size."""
    return image.resize(size)


def normalize_image(image):
    """Normalize image values to range [0, 1]."""
    return np.array(image).astype(np.float32) / 255.0


def resize_normalize(image):
    return resize_image(normalize_image(image))


def is_color_image(image):
    """Check if the image has three channels (RGB)."""
    return image.ndim == 3 and image.shape[2] == 3


def create_colorization_task(image):
    """Create a colorization task by converting the image to grayscale."""
    # Convert image to numpy array if it's not already
    if isinstance(image, tf.Tensor):
        image_array = image.numpy()
    else:
        image_array = np.array(image)

    # Check if the image is already grayscale
    if len(image_array.shape) == 2 or image_array.shape[-1] == 1:
        return image_array

    # Convert RGB to grayscale using the luminosity method
    gray_image = np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])
    return np.expand_dims(gray_image, axis=-1)


def augment_image(image):
    """Apply basic augmentations to the image."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image
