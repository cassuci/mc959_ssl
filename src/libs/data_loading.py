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

def parse_function(filename, label, data_dir):
    """Load image from filename."""
    # Load the image from file
    filename = filename.numpy().decode("utf-8")
    data_dir = data_dir.numpy().decode("utf-8")
    image = np.load(os.path.join(data_dir, "classification", f"{filename}.npy"))
    return image, label


def load_classification_data(data_dir, split_list_file):
    """Create a tf.data.Dataset for the classification task."""
    task_dir = os.path.join(data_dir, "classification")

    # Read data.json file containing image names and corresponding objects
    with open(os.path.join(task_dir, 'data.json'), 'r') as file:
        annotations = json.load(file)

    # Read train.txt or val.txt file containing list of images for each split
    with open(split_list_file) as file:
        split_files = [line.rstrip() for line in file]

    # Create list of (filename, label) pairs
    filenames = []
    labels = []
    for filename in split_files:
        filenames.append(filename)
        labels.append(objects_to_labels(annotations[filename]))

    labels = np.array(labels).astype(np.float32)

    # Create a tf.data.Dataset from filenames and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(lambda filename, label: tf.py_function(
        func=parse_function,
        inp=[filename, label, data_dir],
        Tout=(tf.float32, tf.float32)
    ), num_parallel_calls=tf.data.AUTOTUNE)

    return dataset


def create_dataset(data_dir, split_list_file, batch_size):
    """Load the data and prepare it as a batched tf.data.Dataset."""
    dataset = load_classification_data(data_dir, split_list_file)
    return dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)