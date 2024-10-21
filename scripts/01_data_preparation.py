import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.libs.data_processing import (
    load_image,
    resize_image,
    normalize_image,
    create_inpainting_task,
    create_colorization_task,
)


def is_color_image(image):
    """Check if the image has three channels (RGB)."""
    return image.ndim == 3 and image.shape[2] == 3


def prepare_coco_data(data_dir, output_dir, num_samples=10000):
    """Prepare COCO dataset for pretext tasks."""
    image_dir = os.path.join(data_dir, "train2017")
    output_dir_inpainting = os.path.join(output_dir, "coco", "inpainting")
    output_dir_colorization = os.path.join(output_dir, "coco", "colorization")
    os.makedirs(output_dir_inpainting, exist_ok=True)
    os.makedirs(output_dir_colorization, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    np.random.shuffle(image_files)

    for i, image_file in enumerate(tqdm(image_files[:num_samples], desc="Preparing COCO data")):
        image_path = os.path.join(image_dir, image_file)
        image = load_image(image_path)
        image = resize_image(image)
        image = normalize_image(image)

        # Skip grayscale images
        if not is_color_image(image):
            continue

        # Inpainting task
        masked_image, mask = create_inpainting_task(image)
        np.save(os.path.join(output_dir_inpainting, f"masked_{i}.npy"), masked_image)
        np.save(os.path.join(output_dir_inpainting, f"mask_{i}.npy"), mask)
        np.save(os.path.join(output_dir_inpainting, f"original_{i}.npy"), image)

        # Colorization task
        gray_image = create_colorization_task(image)
        np.save(os.path.join(output_dir_colorization, f"gray_{i}.npy"), gray_image)
        np.save(os.path.join(output_dir_colorization, f"color_{i}.npy"), image)


def prepare_pascal_voc_data(data_dir, output_dir):
    """Prepare Pascal VOC dataset for classification task."""
    image_dir = os.path.join(data_dir, "JPEGImages")
    annotation_dir = os.path.join(data_dir, "Annotations")
    output_dir_classification = os.path.join(output_dir, "pascal_voc", "classification")
    os.makedirs(output_dir_classification, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    for image_file in tqdm(image_files, desc="Preparing Pascal VOC data"):
        image_path = os.path.join(image_dir, image_file)
        image = load_image(image_path)
        image = resize_image(image)
        image = normalize_image(image)

        # Skip grayscale images
        if not is_color_image(image):
            continue

        # Save the preprocessed image
        np.save(os.path.join(output_dir_classification, f"{image_file[:-4]}.npy"), image)

    # TODO: Process annotations and create labels for classification task


if __name__ == "__main__":
    coco_dir = os.path.join("/mnt/f/ssl_images/data", "coco")
    # coco_dir = os.path.join("F:\\ssl_images\\data", "coco")
    pascal_voc_dir = os.path.join("/mnt/f/ssl_images/data", "pascal_voc")
    # pascal_voc_dir = os.path.join("F:\\ssl_images\\data", "pascal_voc")
    output_dir = os.path.join("/mnt/f/ssl_images/data", "processed")

    prepare_coco_data(coco_dir, output_dir)
    prepare_pascal_voc_data(pascal_voc_dir, output_dir)

    print("Data preparation completed successfully!")
