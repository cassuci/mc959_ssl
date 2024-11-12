import os
import sys
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
import cv2
import pycocotools.mask as mask_util

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


def prepare_coco_data(data_dir, output_dir, split="train", num_samples=None):
    """Prepare COCO dataset for pretext tasks."""
    image_dir = os.path.join(data_dir, f"{split}2017")
    annotations_path = os.path.join(data_dir, "annotations", f"instances_{split}2017.json")
    output_dir_segmentation = os.path.join(output_dir, "coco", "segmentation", f"{split}2017")
    os.makedirs(output_dir_segmentation, exist_ok=True)

    # Load the annotations
    try:
        with open(annotations_path, "r") as f:
            annotations = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Failed to load annotations from {annotations_path}")
        return

    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    num_classes = len(set([ann["category_id"] for ann in annotations["annotations"]]))

    for i, image_file in enumerate(
        tqdm(
            image_files[:num_samples],
            desc=f"Preparing COCO data (segmentation) - {split}",
        )
    ):
        image_path = os.path.join(image_dir, image_file)
        image = load_image(image_path)
        image = resize_image(image)
        image = normalize_image(image)

        # Skip grayscale images
        if not is_color_image(image):
            continue

        height, width = image.shape[:2]
        segmentation_mask = np.zeros((height, width, num_classes), dtype=np.uint8)

        for annotation in annotations["annotations"]:
            if annotation["image_id"] == int(image_file.split(".")[0]):
                category_id = annotation["category_id"]
                segmentation = annotation["segmentation"]
                binary_mask = mask_util.decode(
                    {"size": [height, width], "counts": segmentation.encode()}
                )
                segmentation_mask[:, :, category_id - 1] = np.maximum(
                    segmentation_mask[:, :, category_id - 1], binary_mask
                )

        gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        inputgray_image = np.expand_dims(gray_image, axis=-1)

        np.save(os.path.join(output_dir_segmentation, f"image_{i}.npy"), image)
        np.save(os.path.join(output_dir_segmentation, f"inputgray_{i}.npy"), inputgray_image)
        np.save(os.path.join(output_dir_segmentation, f"mask_{i}.npy"), segmentation_mask)


if __name__ == "__main__":
    coco_dir = os.path.join("/mnt/f/ssl_images/data", "coco")
    output_dir = os.path.join("/mnt/f/ssl_images/data", "processed")

    prepare_coco_data(coco_dir, output_dir, "train", num_samples=1000)
    prepare_coco_data(coco_dir, output_dir, "validation", num_samples=200)
    print("Data preparation completed successfully!")
