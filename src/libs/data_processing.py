import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

def load_image(image_path):
    """
    Uses PIL to open an image.

    Args:
        image_path (str): Path to the image.

    Returns:
        PIL.Image: PIL object for the image.
    """
    return Image.open(image_path)


def resize_image(image, size=(224, 224)):
    """
    Resize an image to a given size while keeping the number of channels.

    Args:
        image (PIL.Image or numpy.ndarray): Image to be resized.
        size (tuple): Height and width to be used for resizing. Defaults to (224, 224).

    Returns:
        numpy.ndarray: Resized image.
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure the image is a numpy array
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected PIL.Image or numpy.ndarray, got {type(image)}")
    
    # Resize the image
    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

    return resized_image


def normalize_image(image):
    """
    Normalize image values to range [0, 1].

    Args:
        image (PIL.Image or numpy.ndarray): Image to be normalized.

    Returns:
        numpy.ndarray: Normalized image.
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    return image.astype(np.float32) / 255.0


def resize_normalize(image):
    """
    Resize and normalize image

    Args:
        image (PIL.Image or numpy.ndarray): Image to be resized and normalized.

    Returns:
        numpy.ndarray: Resized and normalized image.
    """
    # First normalize, then resize to ensure consistent processing
    normalized_image = normalize_image(image)
    return resize_image(normalized_image)


def is_color_image(image):
    """
    Check if the image has three channels (RGB).

    Args:
        image (PIL.Image or numpy.ndarray): Input image.

    Returns:
        bool: True if it's a colored image.
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Check dimensionality and number of channels
    return len(image.shape) == 3 and image.shape[2] == 3


def create_colorization_task(image):
    """
    Create a colorization task by converting the image to grayscale.

    Args:
        image (numpy.array, tensorflow.Tensor or alike): Input image.

    Returns:
        numpy.array: Grayscale image for colorization task.
    """
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
    """
    Apply basic augmentations to the image.

    Args:
        image (tensorflow.Tensor): Input image.

    Returns:
        tensorflow.Tensor: Augmented version of image.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image
