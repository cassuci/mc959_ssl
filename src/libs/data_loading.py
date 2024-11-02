import os
import json
import numpy as np
import tensorflow as tf


pascal_voc_classes = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
                      'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
                      'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
pascal_label2int = {label: index for index, label in enumerate(pascal_voc_classes)}
pascal_int2label = {index: label for index, label in enumerate(pascal_voc_classes)}


def objects_to_labels(objects, num_classes=20):
    """Transforms objects to int labels."""
    labels = np.zeros(num_classes)
    for obj in objects:
        assert obj in pascal_voc_classes, f"{obj} not in Pascal VOC classes list"
        labels[pascal_label2int[obj]] = 1
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