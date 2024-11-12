import os
import numpy as np
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

import sys
# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#from src.libs.data_processing import (
    #resize_image,
#)


def resize_image(image, size=(224, 224)):
    return cv2.resize(image, size)

def normalize_image(image):
    """Normalize image values to range [0, 1]."""
    return np.array(image).astype(np.float32) / 255.0

def resize_normalize(image):
    return resize_image(normalize_image(image))

def is_color_image(image):
    """Check if the image has three channels (RGB)."""
    return image.ndim == 3 and image.shape[2] == 3


def prepare_coco_data(data_dir, output_dir, split="train", num_samples=None):
    """Prepare COCO dataset for pretext tasks."""

    annotations_path = os.path.join(
        data_dir, "annotations", f"instances_{split}2017.json"
    )
    output_dir_segmentation = os.path.join(
        output_dir, "coco", "segmentation", f"{split}2017"
    )
    os.makedirs(output_dir_segmentation, exist_ok=True)

    # Initialize COCO API for instance annotations
    coco = COCO(annotations_path)

    # Get the image IDs and annotations
    img_ids = coco.getImgIds()
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=img_ids))

    # You can modify catIds to match the categories you're interested in
    catIds = coco.getCatIds(
        catNms=[
            "person",
            "car",
            "chair",
            "book",
            "bottle",
            "cup",
            "dining table",
            "traffic light",
            "bowl",
            "handbag",
        ]
    )  # Adjust category names as needed
    cat_names = [
        cat["name"] for cat in coco.loadCats(catIds)
    ]  # Get category names for the selected categories

    for i, img_id in enumerate(
        tqdm(img_ids[:num_samples], desc=f"Preparing COCO data - {split}")
    ):
        # Load image data
        img_data = coco.loadImgs(img_id)[0]
        img = io.imread(os.path.join(data_dir, f"{split}2017", img_data["file_name"]))
        height, width = img.shape[:2]

        # Skip grayscale images
        if not is_color_image(img):
            continue

        # Initialize binary mask for all selected classes
        num_classes = len(catIds)  # Number of categories we're interested in
        binary_masks = np.zeros((height, width, num_classes), dtype=np.uint8)

        # Loop through annotations for the current image
        for ann in annotations:
            if ann["image_id"] == img_id:
                # Get the category ID for this annotation
                cat_id = ann["category_id"]

                # Find the class index in catIds
                if cat_id in catIds:
                    class_idx = catIds.index(cat_id)

                    # Create a mask for this annotation using COCO's annToMask function
                    mask = coco.annToMask(ann)

                    # Add the mask to the binary mask for this class (use OR to accumulate masks)
                    binary_masks[:, :, class_idx] = np.maximum(
                        binary_masks[:, :, class_idx], mask
                    )

        # Convert to grayscale
        gray_image = np.dot(
            img[..., :3], [0.2989, 0.5870, 0.1140]
        )  # Standard RGB to grayscale
        input_gray_image = np.expand_dims(gray_image, axis=-1)

        img = resize_normalize(img)
        input_gray_image = resize_normalize(input_gray_image)
        binary_masks = resize_image(binary_masks)

        # Save the images and masks as numpy arrays
        np.save(os.path.join(output_dir_segmentation, f"image_{i}.npy"), img)
        np.save(
            os.path.join(output_dir_segmentation, f"inputgray_{i}.npy"),
            input_gray_image,
        )
        np.save(os.path.join(output_dir_segmentation, f"mask_{i}.npy"), binary_masks)

        # Optional: Display the binary masks for each class
        #for class_idx in range(num_classes):
            #plt.imshow(binary_masks[:, :, class_idx], cmap="gray")
            #plt.title(f"Class: {cat_names[class_idx]}")
            #plt.axis("off")
            #plt.show()


if __name__ == "__main__":
    coco_dir = os.path.join("/mnt/f/ssl_images/data", "coco")
    output_dir = os.path.join("/mnt/f/ssl_images/data", "processed")

    # Run data preparation
    prepare_coco_data(coco_dir, output_dir, "train", num_samples=1000)
    prepare_coco_data(coco_dir, output_dir, "val", num_samples=200)
    print("Data preparation completed successfully!")
