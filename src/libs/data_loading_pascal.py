import os
import json
import numpy as np
import tensorflow as tf

# Pascal VOC Class Definitions
pascal_voc_classes = [
    "person",
    "bird",
    "cat",
    "cow",
    "dog",
    "horse",
    "sheep",
    "aeroplane",
    "bicycle",
    "boat",
    "bus",
    "car",
    "motorbike",
    "train",
    "bottle",
    "chair",
    "diningtable",
    "pottedplant",
    "sofa",
    "tvmonitor",
]
pascal_label2int = {label: index for index, label in enumerate(pascal_voc_classes)}
pascal_int2label = {index: label for index, label in enumerate(pascal_voc_classes)}


def objects_to_labels(objects, num_classes=20):
    """
    Transform list of object names to a binary label vector.

    Args:
        objects (list): List of object names in the image
        num_classes (int): Total number of classes in Pascal VOC

    Returns:
        np.ndarray: Binary vector of class labels
    """
    labels = np.zeros(num_classes)
    for obj in objects:
        assert obj in pascal_voc_classes, f"{obj} not in Pascal VOC classes list"
        labels[pascal_label2int[obj]] = 1
    return labels


def parse_function_classification(filename, label, data_dir, single_channel=False):
    """
    Load and preprocess image for classification task.

    Args:
        filename (tf.Tensor): Image filename
        label (tf.Tensor): Multi-hot label vector
        data_dir (tf.Tensor): Base data directory
        single_channel (bool): Whether to convert to single channel

    Returns:
        Tuple of (image, label) tensors
    """
    # Decode filename and data directory
    filename = filename.numpy().decode("utf-8")
    data_dir = data_dir.numpy().decode("utf-8")

    # Load preprocessed image
    image = np.load(os.path.join(data_dir, "pascal_voc", "classification", f"{filename}.npy"))

    if single_channel:
        # Convert to single channel by weighted grayscale conversion
        image_mean = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        image_mean = np.expand_dims(image_mean, axis=-1)
        return image_mean, label

    return image, label


def load_classification_data(data_dir, split="train", single_channel=False):
    """
    Create a tf.data.Dataset for the Pascal VOC classification task.

    Args:
        data_dir (str): Base directory containing processed data
        split (str): Data split to load ('train', 'val', or 'test')
        single_channel (bool): Whether to convert images to single channel

    Returns:
        tf.data.Dataset: Dataset for classification task
    """
    # Construct full path to the split data JSON
    task_dir = os.path.join(data_dir, "processed", "pascal_voc", "classification")
    split_file = os.path.join(task_dir, f"{split}_data.json")

    # Read annotations
    with open(split_file, "r") as file:
        annotations = json.load(file)

    # Create list of (filename, label) pairs
    filenames = list(annotations.keys())
    labels = [objects_to_labels(annotations[filename]) for filename in filenames]
    labels = np.array(labels).astype(np.float32)

    # Create a tf.data.Dataset from filenames and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(
        lambda filename, label: tf.py_function(
            func=parse_function_classification,
            inp=[filename, label, data_dir, single_channel],
            Tout=(tf.float32, tf.float32),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset


def create_dataset_classification(data_dir, split="train", batch_size=32, single_channel=False):
    """
    Prepare a batched and shuffled dataset for classification.

    Args:
        data_dir (str): Base directory containing processed data
        split (str): Data split to load ('train', 'val', or 'test')
        batch_size (int): Number of images per batch
        single_channel (bool): Whether to convert images to single channel

    Returns:
        tf.data.Dataset: Batched and shuffled classification dataset
    """
    dataset = load_classification_data(data_dir, split, single_channel)
    return dataset.shuffle(500).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def parse_function_segmentation(filename, label, data_dir, single_channel=False):
    """
    Load and preprocess image and mask for segmentation task.

    Args:
        filename (tf.Tensor): Image filename
        label (tf.Tensor): Dummy label (not used in segmentation)
        data_dir (tf.Tensor): Base data directory
        single_channel (bool): Whether to convert image to single channel

    Returns:
        Tuple of (image, mask) tensors
    """
    # Decode filename and data directory
    filename = filename.numpy().decode("utf-8")
    data_dir = data_dir.numpy().decode("utf-8")

    # Load preprocessed image and mask
    image_path = os.path.join(
        data_dir, "processed", "pascal_voc", "segmentation", f"{filename}.npy"
    )
    mask_path = os.path.join(
        data_dir, "processed", "pascal_voc", "segmentation", f"{filename}_mask.npy"
    )

    image = np.load(image_path)
    mask = np.load(mask_path)

    if single_channel:
        # Convert to single channel by weighted grayscale conversion
        image_mean = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        image_mean = np.expand_dims(image_mean, axis=-1)
        return image_mean, mask

    return image, mask


def load_segmentation_data(data_dir, split="train", single_channel=False):
    """
    Create a tf.data.Dataset for the Pascal VOC segmentation task.

    Args:
        data_dir (str): Base directory containing processed data
        split (str): Data split to load ('train', 'val', or 'test')
        single_channel (bool): Whether to convert images to single channel

    Returns:
        tf.data.Dataset: Dataset for segmentation task
    """
    # Construct full path to the split data JSON
    task_dir = os.path.join(data_dir, "processed", "pascal_voc", "segmentation")
    split_file = os.path.join(task_dir, f"{split}_data.json")

    # Read annotations
    with open(split_file, "r") as file:
        annotations = json.load(file)

    # Create list of filenames
    filenames = list(annotations.keys())

    # Limit dataset size for training and validation
    max_samples = 10000 if split == "train" else 2000

    # Truncate if necessary
    filenames = filenames[:max_samples]

    # Create a tf.data.Dataset from filenames
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(
        lambda filename: tf.py_function(
            func=parse_function_segmentation,
            inp=[filename, None, data_dir, single_channel],
            Tout=(tf.float32, tf.float32),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset


def create_dataset_segmentation(data_dir, split="train", batch_size=32, single_channel=False):
    """
    Prepare a batched and shuffled dataset for segmentation.

    Args:
        data_dir (str): Base directory containing processed data
        split (str): Data split to load ('train', 'val', or 'test')
        batch_size (int): Number of images per batch
        single_channel (bool): Whether to convert images to single channel

    Returns:
        tf.data.Dataset: Batched and shuffled segmentation dataset
    """
    dataset = load_segmentation_data(data_dir, split, single_channel)
    return dataset.shuffle(500).batch(batch_size).prefetch(tf.data.AUTOTUNE)
