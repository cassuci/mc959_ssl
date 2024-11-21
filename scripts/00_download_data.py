# scripts/download_datasets.py

import os
import urllib.request
import tarfile
import zipfile
import shutil
from tqdm import tqdm


def download_file(url, filename):
    """Download a file with a progress bar."""
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(
            url, filename, reporthook=lambda b, bsize, tsize: t.update(bsize)
        )


def download_and_extract_coco(data_dir):
    """Download and extract COCO dataset."""
    coco_url = "http://images.cocodataset.org/zips/train2017.zip"
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    os.makedirs(data_dir, exist_ok=True)

    # Download images
    print("Downloading COCO images...")
    download_file(coco_url, os.path.join(data_dir, "train2017.zip"))

    # Download annotations
    print("Downloading COCO annotations...")
    download_file(annotations_url, os.path.join(data_dir, "annotations_trainval2017.zip"))

    # Extract images
    print("Extracting COCO images...")
    with zipfile.ZipFile(os.path.join(data_dir, "train2017.zip"), "r") as zip_ref:
        zip_ref.extractall(data_dir)

    # Extract annotations
    print("Extracting COCO annotations...")
    with zipfile.ZipFile(os.path.join(data_dir, "annotations_trainval2017.zip"), "r") as zip_ref:
        zip_ref.extractall(data_dir)

    # Clean up zip files
    os.remove(os.path.join(data_dir, "train2017.zip"))
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


if __name__ == "__main__":
    coco_dir = os.path.join("data", "coco")
    #coco_dir = os.path.join("F:\\ssl_images\\data", "coco")
    # pascal_voc_dir = os.path.join("/mnt/f/ssl_images/data", "pascal_voc")
    # pascal_voc_dir = os.path.join("data", "pascal_voc")

    download_and_extract_coco(coco_dir)
    #download_and_extract_pascal_voc(pascal_voc_dir)

    print("Datasets downloaded and extracted successfully!")
