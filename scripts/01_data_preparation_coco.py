import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
from skimage.color import rgb2lab
import skimage.io as io
from pycocotools.coco import COCO
import concurrent.futures

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.libs.data_processing import (
    load_image,
    resize_image,
    normalize_image,
    resize_normalize,
    create_colorization_task,
)


def is_color_image(image):
    """
    Determine if an input image is a color image with three RGB channels.

    This function checks the dimensionality and channel count of the image to 
    verify if it's a color image. Color images typically have three dimensions 
    with the last dimension representing color channels.

    Args:
        image (numpy.ndarray): Input image to check for color channels.

    Returns:
        bool: True if the image has three channels, False otherwise.
    """
    return image.ndim == 3 and image.shape[2] == 3


def prepare_coco_data_colorization(data_dir, output_dir, num_samples=None):
    """
    Prepare COCO dataset images for colorization pretext task.

    This function processes color images from the COCO dataset for a colorization 
    task. It performs the following key steps:
    1. Randomly shuffle and select images
    2. Resize and normalize images
    3. Convert images to LAB color space
    4. Create grayscale versions for colorization task
    5. Save preprocessed images as NumPy arrays

    Args:
        data_dir (str): Root directory of the COCO dataset.
        output_dir (str): Directory to save processed colorization data.
        num_samples (int, optional): Number of images to process. 
                                     Defaults to processing all images.

    Notes:
        - Skips grayscale images
        - Saves grayscale and color (LAB) images as separate NumPy arrays
    """
    image_dir = os.path.join(data_dir, "train2017")
    output_dir_colorization = os.path.join(output_dir, "coco", "colorization")
    os.makedirs(output_dir_colorization, exist_ok=True)

    # Get all JPEG image files and shuffle them
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    np.random.shuffle(image_files)

    if num_samples is None:
        num_samples = len(image_files)

    for i, image_file in enumerate(
        tqdm(image_files[:num_samples], desc="Preparing COCO data")
    ):
        image_path = os.path.join(image_dir, image_file)
        image = load_image(image_path)
        image = resize_image(image)
        image = normalize_image(image)

        # Skip grayscale images
        if not is_color_image(image):
            continue

        # Colorization task in LAB color space
        lab_image = rgb2lab(image)  # Convert RGB to LAB
        gray_image = create_colorization_task(image)
        
        # Save preprocessed images
        np.save(os.path.join(output_dir_colorization, f"gray{i}.npy"), gray_image)
        np.save(
            os.path.join(output_dir_colorization, f"color{i}.npy"), lab_image
        )  # Save LAB image


def save_segmentation_arrays(
    coco, data_dir, output_dir_segmentation, img_id, catIds, split
):
    """
    Process a single image for semantic segmentation task.

    Generates binary masks for specified object categories, including a background mask.
    Handles image resizing, normalization, and mask generation.

    Args:
        coco (COCO): COCO annotations object.
        data_dir (str): Root directory of the COCO dataset.
        output_dir_segmentation (str): Output directory for segmentation data.
        img_id (int): Image ID from COCO dataset.
        catIds (list): List of category IDs to extract segmentation masks for.
        split (str): Dataset split (e.g., 'train', 'val').

    Returns:
        None: Saves processed image, grayscale image, and segmentation masks as NumPy arrays.

    Notes:
        - Skips images without annotations for specified categories
        - Generates multi-channel binary masks 
        - Adds a background mask as the final channel
    """
    img_data = coco.loadImgs(img_id)[0]
    img = io.imread(os.path.join(data_dir, f"{split}2017", img_data["file_name"]))
    height, width = img.shape[:2]

    if not is_color_image(img):
        return

    num_classes = len(catIds)
    binary_masks = np.zeros((height, width, num_classes), dtype=np.uint8)

    # Check if any annotation category is in the specified categories
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
    contains_class = any(ann["category_id"] in catIds for ann in annotations)
    if not contains_class:
        return

    # Generate binary masks for specified categories
    for ann in annotations:
        cat_id = ann["category_id"]
        if cat_id in catIds:
            class_idx = catIds.index(cat_id)
            mask = coco.annToMask(ann)
            binary_masks[:, :, class_idx] = np.maximum(
                binary_masks[:, :, class_idx], mask
            )

    # Compute background mask
    background_mask = np.ones((height, width), dtype=np.float16)
    background_mask = np.maximum(background_mask - np.sum(binary_masks, axis=-1), 0)

    # Add background mask as the last channel
    binary_masks = np.concatenate(
        (binary_masks, np.expand_dims(background_mask, axis=-1)), axis=-1
    )

    # Normalize and convert image to grayscale
    img = resize_normalize(img)
    input_gray_image = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    binary_masks = resize_image(binary_masks)

    # Save processed data
    np.save(os.path.join(output_dir_segmentation, f"image_{img_id}.npy"), img)
    np.save(
        os.path.join(output_dir_segmentation, f"inputgray_{img_id}.npy"),
        input_gray_image,
    )
    np.save(os.path.join(output_dir_segmentation, f"mask_{img_id}.npy"), binary_masks)


def prepare_coco_data_segmentation(
    data_dir, output_dir, split="train", num_samples=None
):
    """
    Prepare COCO dataset for semantic segmentation task.

    Processes images from specified dataset split (train/val) and generates 
    segmentation masks for predefined object categories.

    Args:
        data_dir (str): Root directory of the COCO dataset.
        output_dir (str): Directory to save processed segmentation data.
        split (str, optional): Dataset split to process. Defaults to 'train'.
        num_samples (int, optional): Number of images to process. 
                                     Defaults to processing all images.

    Notes:
        - Uses concurrent processing for efficient data preparation
        - Focuses on specific object categories (person, car, chair, etc.)
        - Generates multi-channel binary masks with background
    """
    annotations_path = os.path.join(
        data_dir, "annotations", f"instances_{split}2017.json"
    )
    output_dir_segmentation = os.path.join(
        output_dir, "coco", "segmentation", f"{split}2017"
    )
    os.makedirs(output_dir_segmentation, exist_ok=True)

    coco = COCO(annotations_path)
    img_ids = coco.getImgIds()
    if num_samples is not None:
        img_ids = img_ids[:num_samples]

    # Define categories for segmentation
    catIds = coco.getCatIds(
        catNms=[
            "person",
            "car",
            "chair",
            # Commented out additional categories for flexibility
            # "book", "bottle", "cup", "dining table", 
            # "traffic light", "bowl", "handbag"
        ]
    )
    cat_names = [cat["name"] for cat in coco.loadCats(catIds)]

    # Use ThreadPoolExecutor for parallel processing of images
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(
                    lambda img_id: save_segmentation_arrays(
                        coco, data_dir, output_dir_segmentation, img_id, catIds, split
                    ),
                    img_ids,
                ),
                total=len(img_ids),
                desc=f"Preparing COCO data - {split}",
            )
        )


def main(
    coco_dir,
    output_dir,
    colorization_samples,
    segmentation_train_samples,
    segmentation_val_samples,
):
    """
    Main entry point for COCO dataset preprocessing.

    Orchestrates two primary preprocessing tasks:
    1. Image Colorization: Prepare grayscale and color image pairs
    2. Semantic Segmentation: Generate multi-class segmentation masks

    Args:
        coco_dir (str): Directory containing the COCO dataset.
        output_dir (str): Directory to save processed data.
        colorization_samples (int): Number of images for colorization task.
        segmentation_train_samples (int): Number of training images for segmentation.
        segmentation_val_samples (int): Number of validation images for segmentation.
    """
    prepare_coco_data_colorization(
        coco_dir, output_dir, num_samples=colorization_samples
    )

    prepare_coco_data_segmentation(
        coco_dir, output_dir, "train", num_samples=segmentation_train_samples
    )
    prepare_coco_data_segmentation(
        coco_dir, output_dir, "val", num_samples=segmentation_val_samples
    )

    print("Data preparation completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess COCO dataset for colorization and segmentation tasks. "
        "Generates preprocessed data for machine learning model training."
    )
    parser.add_argument(
        "--coco_dir",
        type=str,
        default=os.path.join("data", "coco"),
        help="Directory containing the COCO dataset. Default: 'data/coco'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("data", "processed"),
        help="Directory to store processed data. Default: 'data/processed'.",
    )
    parser.add_argument(
        "--colorization_samples",
        type=int,
        default=1000,
        help="Number of images for colorization task preprocessing. Default: 1000.",
    )
    parser.add_argument(
        "--segmentation_train_samples",
        type=int,
        default=1000,
        help="Number of training images for segmentation preprocessing. Default: 1000.",
    )
    parser.add_argument(
        "--segmentation_val_samples",
        type=int,
        default=200,
        help="Number of validation images for segmentation preprocessing. Default: 200.",
    )

    args = parser.parse_args()
    main(
        coco_dir=args.coco_dir,
        output_dir=args.output_dir,
        colorization_samples=args.colorization_samples,
        segmentation_train_samples=args.segmentation_train_samples,
        segmentation_val_samples=args.segmentation_val_samples,
    )