import os
import sys
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
from skimage.color import rgb2lab, lab2rgb  # Import for color space conversion
import xml.etree.ElementTree as ET

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.libs.data_processing import (
    load_image,
    resize_image,
    normalize_image,
    create_inpainting_task,
    create_colorization_task,
    create_segmentation_task,
)


def is_color_image(image):
    """Check if the image has three channels (RGB)."""
    return image.ndim == 3 and image.shape[2] == 3


def prepare_coco_data(data_dir, output_dir, num_samples=None):
    """Prepare COCO dataset for pretext tasks."""
    image_dir = os.path.join(data_dir, "train2017")
    annotations_dir = os.path.join(data_dir, "annotations")
    output_dir_inpainting = os.path.join(output_dir, "coco", "inpainting")
    output_dir_colorization = os.path.join(output_dir, "coco", "colorization")
    os.makedirs(output_dir_inpainting, exist_ok=True)
    os.makedirs(output_dir_colorization, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    np.random.shuffle(image_files)

    if num_samples is None:
        num_samples = len(image_files)

    for i, image_file in enumerate(tqdm(image_files[:num_samples], desc="Preparing COCO data")):
        image_path = os.path.join(image_dir, image_file)
        image = load_image(image_path)
        image = resize_image(image)
        image = normalize_image(image)

        # Skip grayscale images
        if not is_color_image(image):
            continue

        # Inpainting task
        # masked_image, mask = create_inpainting_task(image)
        # np.save(os.path.join(output_dir_inpainting, f"masked{i}.npy"), masked_image)
        # np.save(os.path.join(output_dir_inpainting, f"mask{i}.npy"), mask)
        # np.save(os.path.join(output_dir_inpainting, f"original{i}.npy"), image)

        # Colorization task in LAB color space
        lab_image = rgb2lab(image)  # Convert RGB to LAB
        gray_image = create_colorization_task(image)
        np.save(os.path.join(output_dir_colorization, f"gray{i}.npy"), gray_image)
        np.save(os.path.join(output_dir_colorization, f"color{i}.npy"), lab_image)  # Save LAB image


def get_pascal_voc_annotations(xml_file):
    """Read XML files to extract list of objects in image."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = list(set([obj.find("name").text for obj in root.findall("object")]))
    return objects


def prepare_pascal_voc_data(data_dir, output_dir):
    """Prepare Pascal VOC dataset for classification task."""
    image_dir = os.path.join(data_dir, "JPEGImages")
    annotation_dir = os.path.join(data_dir, "Annotations")
    output_dir_classification = os.path.join(output_dir, "pascal_voc", "classification")
    os.makedirs(output_dir_classification, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    annotations = dict()

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

        # Create dict with filename and the objects that image contains
        objects = get_pascal_voc_annotations(
            os.path.join(annotation_dir, image_file.replace(".jpg", ".xml"))
        )
        annotations.update({image_file[:-4]: objects})

    with open(os.path.join(output_dir_classification, "data.json"), "w") as file:
        json.dump(annotations, file)


if __name__ == "__main__":
    coco_dir = os.path.join("/mnt/f/ssl_images/data", "coco")
    pascal_voc_dir = os.path.join("/mnt/f/ssl_images/data", "pascal_voc")
    output_dir = os.path.join("/mnt/f/ssl_images/data", "processed")

    prepare_coco_data(coco_dir, output_dir)
    # prepare_pascal_voc_data(pascal_voc_dir, output_dir)
    print("Data preparation completed successfully!")
