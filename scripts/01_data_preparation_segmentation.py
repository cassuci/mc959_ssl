import os
import numpy as np
from pycocotools.coco import COCO
import skimage.io as io
from tqdm import tqdm
import cv2
import concurrent.futures


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


def process_image(coco, data_dir, output_dir_segmentation, img_id, catIds, split):
    img_data = coco.loadImgs(img_id)[0]
    img = io.imread(os.path.join(data_dir, f"{split}2017", img_data["file_name"]))
    height, width = img.shape[:2]

    if not is_color_image(img):
        return

    num_classes = len(catIds)
    binary_masks = np.zeros((height, width, num_classes), dtype=np.uint8)

    annotations = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
    for ann in annotations:
        cat_id = ann["category_id"]
        if cat_id in catIds:
            class_idx = catIds.index(cat_id)
            mask = coco.annToMask(ann)
            binary_masks[:, :, class_idx] = np.maximum(binary_masks[:, :, class_idx], mask)

    gray_image = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    input_gray_image = np.expand_dims(gray_image, axis=-1)

    img = resize_normalize(img)
    input_gray_image = resize_normalize(input_gray_image)
    binary_masks = resize_image(binary_masks)

    np.save(os.path.join(output_dir_segmentation, f"image_{img_id}.npy"), img)
    np.save(os.path.join(output_dir_segmentation, f"inputgray_{img_id}.npy"), input_gray_image)
    np.save(os.path.join(output_dir_segmentation, f"mask_{img_id}.npy"), binary_masks)


def prepare_coco_data(data_dir, output_dir, split="train", num_samples=None):
    annotations_path = os.path.join(data_dir, "annotations", f"instances_{split}2017.json")
    output_dir_segmentation = os.path.join(output_dir, "coco", "segmentation", f"{split}2017")
    os.makedirs(output_dir_segmentation, exist_ok=True)

    coco = COCO(annotations_path)
    img_ids = coco.getImgIds()
    if num_samples is not None:
        img_ids = img_ids[:num_samples]

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
    )
    cat_names = [cat["name"] for cat in coco.loadCats(catIds)]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(
                    lambda img_id: process_image(
                        coco, data_dir, output_dir_segmentation, img_id, catIds, split
                    ),
                    img_ids,
                ),
                total=len(img_ids),
                desc=f"Preparing COCO data - {split}",
            )
        )


if __name__ == "__main__":
    coco_dir = os.path.join("/mnt/f/ssl_images/data", "coco")
    output_dir = os.path.join("/mnt/f/ssl_images/data", "processed")

    prepare_coco_data(coco_dir, output_dir, "train", num_samples=None)
    prepare_coco_data(coco_dir, output_dir, "val", num_samples=None)
    print("Data preparation completed successfully!")
