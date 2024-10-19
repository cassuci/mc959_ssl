# src/libs/data_processing.py

import numpy as np
from PIL import Image
import tensorflow as tf

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
    mask = np.ones_like(image)
    h, w, _ = image.shape
    y = np.random.randint(0, h - mask_size)
    x = np.random.randint(0, w - mask_size)
    mask[y:y+mask_size, x:x+mask_size, :] = 0
    masked_image = image * mask
    return masked_image, mask

def create_colorization_task(image):
    """Create a colorization task by converting the image to grayscale."""
    gray_image = tf.image.rgb_to_grayscale(image)
    return gray_image

def augment_image(image):
    """Apply basic augmentations to the image."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image

def prepare_coco_dataset(image_paths, batch_size=32):
    """Prepare COCO dataset for training."""
    def load_and_preprocess(image_path):
        image = load_image(image_path)
        image = resize_image(image)
        image = normalize_image(image)
        image = augment_image(image)
        return image

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def prepare_pascal_voc_dataset(image_paths, labels, batch_size=32):
    """Prepare Pascal VOC dataset for training."""
    def load_and_preprocess(image_path, label):
        image = load_image(image_path)
        image = resize_image(image)
        image = normalize_image(image)
        image = augment_image(image)
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset