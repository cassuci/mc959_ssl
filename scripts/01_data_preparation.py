# scripts/01_data_preparation.py

import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.libs.data_processing import load_image, resize_image, normalize_image, create_inpainting_task, create_colorization_task

def prepare_coco_dataset(data_dir, output_dir, num_samples=10000):
    """Prepare COCO dataset for pretext tasks."""
    image_dir = os.path.join(data_dir, 'train2017')
    output_inpainting_dir = os.path.join(output_dir, 'inpainting')
    output_colorization_dir = os.path.join(output_dir, 'colorization')

    os.makedirs(output_inpainting_dir, exist_ok=True)
    os.makedirs(output_colorization_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    np.random.shuffle(image_files)

    for i, img_file in enumerate(tqdm(image_files[:num_samples], desc="Preparing COCO dataset")):
        img_path = os.path.join(image_dir, img_file)
        img = load_image(img_path)
        img = resize_image(img)
        img = normalize_image(img)

        # Inpainting task
        masked_img, mask = create_inpainting_task(img)
        np.save(os.path.join(output_inpainting_dir, f'masked_{i}.npy'), masked_img)
        np.save(os.path.join(output_inpainting_dir, f'mask_{i}.npy'), mask)

        # Colorization task
        gray_img = create_colorization_task(img)
        np.save(os.path.join(output_colorization_dir, f'gray_{i}.npy'), gray_img)

        # Save original image for both tasks
        np.save(os.path.join(output_inpainting_dir, f'original_{i}.npy'), img)
        np.save(os.path.join(output_colorization_dir, f'original_{i}.npy'), img)

def prepare_pascal_voc_dataset(data_dir, output_dir):
    """Prepare Pascal VOC dataset for classification."""
    image_dir = os.path.join(data_dir, 'JPEGImages')
    annotation_dir = os.path.join(data_dir, 'Annotations')
    output_image_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    for img_file in tqdm(image_files, desc="Preparing Pascal VOC dataset"):
        img_path = os.path.join(image_dir, img_file)
        img = load_image(img_path)
        img = resize_image(img)
        img = normalize_image(img)

        # Save processed image
        np.save(os.path.join(output_image_dir, f'{os.path.splitext(img_file)[0]}.npy'), img)

        # Process and save label (you might need to implement this based on your specific needs)
        # For simplicity, we're just copying the XML annotation file
        annotation_file = os.path.join(annotation_dir, f'{os.path.splitext(img_file)[0]}.xml')
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f_in, open(os.path.join(output_label_dir, f'{os.path.splitext(img_file)[0]}.xml'), 'w') as f_out:
                f_out.write(f_in.read())

if __name__ == "__main__":
    coco_data_dir = os.path.join('data', 'coco')
    coco_output_dir = os.path.join('data', 'processed', 'coco')
    pascal_voc_data_dir = os.path.join('data', 'pascal_voc')
    pascal_voc_output_dir = os.path.join('data', 'processed', 'pascal_voc')

    prepare_coco_dataset(coco_data_dir, coco_output_dir)
    prepare_pascal_voc_dataset(pascal_voc_data_dir, pascal_voc_output_dir)

    print("Data preparation completed successfully!")