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


def create_inpainting_task(image, mask_size=50):
    """Create an inpainting task by masking a portion of the image."""
    # Convert image to numpy array if it's not already
    image_array = np.array(image)

    # Handle both RGB and grayscale images
    if len(image_array.shape) == 2:
        image_array = np.expand_dims(image_array, axis=-1)

    mask = np.ones_like(image_array)
    h, w, c = image_array.shape
    y = np.random.randint(0, h - mask_size)
    x = np.random.randint(0, w - mask_size)
    mask[y : y + mask_size, x : x + mask_size, :] = 0
    masked_image = image_array * mask
    return masked_image, mask


def create_segmentation_task(image, annotations):
    """
    Create a segmentation task by generating a multi-class segmentation mask for the input image.

    Parameters:
    image_path (str): The file path to the input image.
    annotations_path (str): The file path to the COCO annotations JSON file.

    Returns:
    numpy.ndarray: The original image.
    numpy.ndarray: The multi-class segmentation mask with shape (height, width, num_classes).
    """

    # Create the segmentation mask
    height, width = image.shape[:2]
    num_classes = len(set([ann["category_id"] for ann in annotations["annotations"]]))
    segmentation_mask = np.zeros((height, width, num_classes), dtype=np.uint8)

    for annotation in annotations["annotations"]:
        category_id = annotation["category_id"]
        segmentation = annotation["segmentation"]
        binary_mask = mask_util.decode({"size": [height, width], "counts": segmentation.encode()})
        segmentation_mask[:, :, category_id - 1] = np.maximum(
            segmentation_mask[:, :, category_id - 1], binary_mask
        )

    return segmentation_mask


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
