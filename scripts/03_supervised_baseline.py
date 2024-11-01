# scripts/05_baseline.py

import os
import sys
import json
import numpy as np
import tensorflow as tf

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.resnet import ResNet18

# TODO Create separate file for pascal voc dataloader
pascal_voc_classes = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
                      'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
                      'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
label2int = {label: index for index, label in enumerate(pascal_voc_classes)}
int2label = {index: label for index, label in enumerate(pascal_voc_classes)}


def objects_to_labels(objects, num_classes=20):
    """Transforms objects to int labels."""
    labels = np.zeros(num_classes)
    for obj in objects:
        assert obj in pascal_voc_classes, f"{obj} not in Pascal VOC classes list"
        labels[label2int[obj]] = 1
    return labels


def load_classification_data(data_dir, split_list_file):
    """Load preprocessed data for the classification task."""
    task_dir = os.path.join(data_dir, "classification")

    # Read data.json file containing image names and its corresponding objects
    with open(os.path.join(task_dir, 'data.json'), 'r') as file:
        annotations = json.load(file)

    # Read train.txt or val.txt file containing list of images for each split
    with open(split_list_file) as file:
        split_files = [line.rstrip() for line in file]

    images = []
    labels = []
    # For each input in the split, load preprocessed image and its labels
    for filename in split_files:
        objects = annotations[filename]
        images.append(np.load(os.path.join(task_dir, f'{filename}.npy')))
        labels.append(objects_to_labels(objects, num_classes=len(pascal_voc_classes)))

    return np.array(images), np.array(labels)


def create_dataset(images, labels, batch_size):
    """Create a TensorFlow dataset from images and labels."""
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def fine_tune_model(model, train_dataset, val_dataset, epochs=10):
    """Fine-tune the model on the classification task."""

    # TODO Fix metrics and loss for multilabel classification
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # TODO Add callbacks

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, verbose=1)
    return model, history


if __name__ == "__main__":
    # TODO Customize paths
    data_dir = os.path.join("ssl_images/data", "processed", "pascal_voc")
    metadata_dir = os.path.join("ssl_images/data", "pascal_voc", "ImageSets", "Main")

    model = ResNet18((224, 224, 3), mode='classification')
    print(model.summary())

    # Load classification data
    print("Loading data and creating dataset...")
    train_images, train_labels = load_classification_data(data_dir, split_list_file=os.path.join(metadata_dir, 'train.txt'))
    val_images, val_labels = load_classification_data(data_dir, split_list_file=os.path.join(metadata_dir, 'val.txt'))

    # Create dataset
    train_dataset = create_dataset(train_images, train_labels, batch_size=32)
    val_dataset = create_dataset(val_images, val_labels, batch_size=32)

    # Fine-tune the model
    print("Fine-tuning the model...")
    fine_tuned_model, history = fine_tune_model(model, train_dataset, val_dataset)

    # Save the fine-tuned model
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "baseline_resnet18.h5")
    fine_tuned_model.save_weights(save_path)
    print(f"Final model saved to {save_path}")

    print("Supervised fine-tuning completed successfully!")