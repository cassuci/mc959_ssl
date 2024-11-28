# scripts/download_datasets.py

import os
import urllib.request
import tarfile
import zipfile
import shutil
import argparse
from tqdm import tqdm


def download_file(url, filename):
    """Download a file with a progress bar."""
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(
            url, filename, reporthook=lambda b, bsize, tsize: t.update(bsize)
        )


def download_and_extract_coco(data_dir):
    """Download and extract COCO dataset."""
    train_url = "http://images.cocodataset.org/zips/train2017.zip"
    val_url = "http://images.cocodataset.org/zips/val2017.zip"
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    os.makedirs(data_dir, exist_ok=True)

    # Download train images
    print("Downloading COCO train images...")
    download_file(train_url, os.path.join(data_dir, "train2017.zip"))

    # Download val images
    print("Downloading COCO validation images...")
    download_file(val_url, os.path.join(data_dir, "val2017.zip"))

    # Download annotations
    print("Downloading COCO annotations...")
    download_file(annotations_url, os.path.join(data_dir, "annotations_trainval2017.zip"))

    # Extract train images
    print("Extracting COCO train images...")
    with zipfile.ZipFile(os.path.join(data_dir, "train2017.zip"), "r") as zip_ref:
        zip_ref.extractall(data_dir)

    # Extract val images
    print("Extracting COCO validation images...")
    with zipfile.ZipFile(os.path.join(data_dir, "val2017.zip"), "r") as zip_ref:
        zip_ref.extractall(data_dir)

    # Extract annotations
    print("Extracting COCO annotations...")
    with zipfile.ZipFile(os.path.join(data_dir, "annotations_trainval2017.zip"), "r") as zip_ref:
        zip_ref.extractall(data_dir)

    # Clean up zip files
    os.remove(os.path.join(data_dir, "train2017.zip"))
    os.remove(os.path.join(data_dir, "val2017.zip"))
    os.remove(os.path.join(data_dir, "annotations_trainval2017.zip"))


def download_and_extract_pascal_voc(data_dir):
    """Download and extract Pascal VOC dataset."""
    pascal_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"

    os.makedirs(data_dir, exist_ok=True)

    # Download dataset
    print("Downloading Pascal VOC dataset...")
    download_file(pascal_url, os.path.join(data_dir, "VOCtrainval_11-May-2012.tar"))

    # Extract dataset
    print("Extracting Pascal VOC dataset...")
    with tarfile.open(os.path.join(data_dir, "VOCtrainval_11-May-2012.tar"), "r") as tar_ref:
        tar_ref.extractall(data_dir)

    # Move contents to the main directory and remove the extra folder
    voc_dir = os.path.join(data_dir, "VOCdevkit", "VOC2012")
    for item in os.listdir(voc_dir):
        shutil.move(os.path.join(voc_dir, item), data_dir)
    shutil.rmtree(os.path.join(data_dir, "VOCdevkit"))

    # Clean up tar file
    os.remove(os.path.join(data_dir, "VOCtrainval_11-May-2012.tar"))


def main(coco_dir, pascal_voc_dir):
    """Download and extract datasets."""
    download_and_extract_coco(coco_dir)
    download_and_extract_pascal_voc(pascal_voc_dir)
    print("Datasets downloaded and extracted successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract datasets.")
    parser.add_argument(
        "--coco_dir",
        type=str,
        default=os.path.join("data", "coco"),
        help="Directory to download and extract COCO dataset. Default: 'data/coco'.",
    )
    parser.add_argument(
        "--pascal_voc_dir",
        type=str,
        default=os.path.join("data", "pascal_voc"),
        help="Directory to download and extract Pascal VOC dataset. Default: 'data/pascal_voc'.",
    )

    args = parser.parse_args()
    main(coco_dir=args.coco_dir, pascal_voc_dir=args.pascal_voc_dir)
