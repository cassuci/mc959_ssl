import os
import sys
import json
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
from typing import Dict, List
import random
import argparse
import cv2

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.libs.data_processing import (
    load_image,
    resize_image,
    normalize_image,
)


def is_color_image(image):
    """Check if the image has three channels (RGB)."""
    return image.ndim == 3 and image.shape[2] == 3


def get_pascal_voc_object_classes(xml_file):
    """
    Read XML files to extract all objects with their details.

    Returns:
        List of dictionaries with object details
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bndbox = obj.find("bndbox")

        # Convert to float first, then to int (to handle possible floats in bounding boxes)
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))

        objects.append({"name": name, "bbox": [xmin, ymin, xmax, ymax]})

    return objects


def load_and_create_multi_channel_mask(mask_dir, image_file, classes):
    """
    Create a multi-channel mask where each channel represents a different class.

    Args:
        mask_dir (str): Directory containing segmentation masks
        image_file (str): Name of the image file
        classes (list): List of unique classes in the annotation

    Returns:
        np.ndarray: Multi-channel mask with shape (height, width, num_classes)
    """
    # Create color to class mapping (VOC standard color map)
    color_map = {
        "background": [0, 0, 0],
        "aeroplane": [128, 0, 0],
        "bicycle": [0, 128, 0],
        "bird": [128, 128, 0],
        "boat": [0, 0, 128],
        "bottle": [128, 0, 128],
        "bus": [0, 128, 128],
        "car": [128, 128, 128],
        "cat": [64, 0, 0],
        "chair": [192, 0, 0],
        "cow": [64, 128, 0],
        "diningtable": [192, 128, 0],
        "dog": [64, 0, 128],
        "horse": [192, 0, 128],
        "motorbike": [64, 128, 128],
        "person": [192, 128, 128],
        "pottedplant": [0, 64, 0],
        "sheep": [128, 64, 0],
        "sofa": [0, 192, 0],
        "train": [128, 192, 0],
        "tvmonitor": [0, 64, 128],
    }

    # Load original mask (in color)
    mask_path = os.path.join(mask_dir, image_file.replace(".jpg", ".png"))
    if not os.path.exists(mask_path):
        return None

    # Read mask in color
    original_mask = cv2.imread(mask_path)

    # Resize mask
    original_mask = cv2.resize(original_mask, (224, 224), interpolation=cv2.INTER_NEAREST)

    # Create multi-channel mask
    multi_channel_mask = np.zeros((224, 224, len(color_map)), dtype=np.uint8)

    # For each class, create a binary mask by color matching
    for class_name, color in color_map.items():
        # Create a mask for each class by comparing exact color values
        class_mask = np.all(original_mask == color, axis=-1)
        multi_channel_mask[:, :, list(color_map.keys()).index(class_name)] = class_mask.astype(
            np.uint8
        )

    return multi_channel_mask


def split_dataset(
    data: Dict[str, List[str]],
    test_split: float = 0.2,
    train_val_split: float = 0.8,
    seed: int = 42,
):
    """
    Split dataset into train, validation, and test sets.

    Args:
        data (Dict): Dictionary of image annotations
        test_split (float): Proportion of validation set to use as test set
        train_val_split (float): Proportion of remaining data to use for training
        seed (int): Random seed for reproducibility

    Returns:
        Tuple of dictionaries: (train_data, val_data, test_data)
    """
    random.seed(seed)
    image_names = list(data.keys())
    random.shuffle(image_names)

    # Split into train+val and test sets
    split_idx = int(len(image_names) * (1 - test_split))
    train_val_images = image_names[:split_idx]
    test_images = image_names[split_idx:]

    # Further split train+val into train and validation
    train_split_idx = int(len(train_val_images) * train_val_split)
    train_images = train_val_images[:train_split_idx]
    val_images = train_val_images[train_split_idx:]

    # Create new dictionaries for each split
    train_data = {img: data[img] for img in train_images}
    val_data = {img: data[img] for img in val_images}
    test_data = {img: data[img] for img in test_images}

    return train_data, val_data, test_data


def prepare_pascal_voc_data(data_dir, output_dir, task="segmentation"):
    """
    Prepare Pascal VOC dataset for classification or segmentation task.

    Args:
        data_dir (str): Path to Pascal VOC dataset
        output_dir (str): Path to save processed data
        task (str): 'classification' or 'segmentation'
    """
    image_dir = os.path.join(data_dir, "JPEGImages")
    annotation_dir = os.path.join(data_dir, "Annotations")
    segmentation_dir = os.path.join(data_dir, "SegmentationClass")
    output_dir_processed = os.path.join(output_dir, "pascal_voc", task)
    os.makedirs(output_dir_processed, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    annotations = {}

    for image_file in tqdm(image_files, desc=f"Preparing Pascal VOC data for {task}"):
        image_path = os.path.join(image_dir, image_file)
        image = load_image(image_path)
        image = resize_image(image)
        image = normalize_image(image)

        # Skip grayscale images
        if not is_color_image(image):
            continue

        # Save the preprocessed image
        np.save(os.path.join(output_dir_processed, f"{image_file[:-4]}.npy"), image)

        # Extract annotations based on task
        if task == "classification":
            objects = get_pascal_voc_object_classes(
                os.path.join(annotation_dir, image_file.replace(".jpg", ".xml"))
            )
            # For classification, store just the class names
            object_names = [obj["name"] for obj in objects]
        elif task == "segmentation":
            # Load and save multi-channel segmentation mask
            mask = load_and_create_multi_channel_mask(
                segmentation_dir,
                image_file,
                get_pascal_voc_object_classes(
                    os.path.join(annotation_dir, image_file.replace(".jpg", ".xml"))
                ),
            )

            if mask is not None:
                np.save(os.path.join(output_dir_processed, f"{image_file[:-4]}_mask.npy"), mask)
                # Get object names for metadata
                objects = get_pascal_voc_object_classes(
                    os.path.join(annotation_dir, image_file.replace(".jpg", ".xml"))
                )
                object_names = [obj["name"] for obj in objects]
            else:
                continue  # Skip if no mask found

        annotations[image_file[:-4]] = object_names

    # Split dataset
    train_data, val_data, test_data = split_dataset(annotations)

    # Save splits
    splits = {"train": train_data, "val": val_data, "test": test_data}

    for split_name, split_data in splits.items():
        with open(os.path.join(output_dir_processed, f"{split_name}_data.json"), "w") as file:
            json.dump(split_data, file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prepare Pascal VOC dataset for various tasks.")
    parser.add_argument(
        "--pascal_voc_dir",
        type=str,
        default=os.path.join("data", "pascal_voc"),
        help="Directory where the Pascal Voc dataset is stored. Default: 'data/pascal_voc'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("data", "processed"),
        help="Directory to store the processed data. Default: 'data/processed'.",
    )
    args = parser.parse_args()

    # Prepare classification data
    prepare_pascal_voc_data(args.pascal_voc_dir, args.output_dir, task='classification')

    # Prepare segmentation data
    # We discontinued the idea of using Pascal VOC for segmentation, so this is not used
    #prepare_pascal_voc_data(args.pascal_voc_dir, args.output_dir, task="segmentation")

    print("Data preparation completed successfully!")
