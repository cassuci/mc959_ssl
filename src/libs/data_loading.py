import os
import json
import numpy as np
import tensorflow as tf

# <------- Classification data loader functions ------->

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
    """Transforms objects to int labels."""
    labels = np.zeros(num_classes)
    for obj in objects:
        assert obj in pascal_voc_classes, f"{obj} not in Pascal VOC classes list"
        labels[pascal_label2int[obj]] = 1
    return labels


def parse_function_classification(filename, label, data_dir, single_channel=False):
    """Load image from filename and average all 3 channels into a single channel."""
    # Load the image from file
    filename = filename.numpy().decode("utf-8")
    data_dir = data_dir.numpy().decode("utf-8")
    image = np.load(os.path.join(data_dir, "classification", f"{filename}.npy"))

    if single_channel:
        # Average the three channels
        # image_mean = np.mean(image, axis=-1)  # Average across the last dimension (channels)
        image_mean = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

        # Expand dimensions to make it (height, width, 1)
        image_mean = np.expand_dims(image_mean, axis=-1)  # Add a new axis for the single channel

        return image_mean, label

    return image, label


def load_classification_data(data_dir, split_list_file, single_channel=False):
    """Create a tf.data.Dataset for the classification task."""
    task_dir = os.path.join(data_dir, "classification")

    # Read data.json file containing image names and corresponding objects
    with open(os.path.join(task_dir, "data.json"), "r") as file:
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
    dataset = dataset.map(
        lambda filename, label: tf.py_function(
            func=parse_function_classification,
            inp=[filename, label, data_dir, single_channel],
            Tout=(tf.float32, tf.float32),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset


def create_dataset_classification(data_dir, split_list_file, batch_size, single_channel=False):
    """Load the data and prepare it as a batched tf.data.Dataset."""
    dataset = load_classification_data(data_dir, split_list_file, single_channel)
    return dataset.shuffle(500).batch(batch_size).prefetch(tf.data.AUTOTUNE)


# <------- Segmentation data loader functions ------->


def parse_function_segmentation(image_path, mask_path, single_channel):
    """Load images and masks."""
    # Load the image from file
    image_path = image_path.numpy().decode("utf-8")
    mask_path = mask_path.numpy().decode("utf-8")

    image = np.load(image_path)
    mask = np.load(mask_path)

    if single_channel:
        # Average the three channels
        # image_mean = np.mean(image, axis=-1)  # Average across the last dimension (channels)
        image_mean = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

        # Expand dimensions to make it (height, width, 1)
        image_mean = np.expand_dims(image_mean, axis=-1)  # Add a new axis for the single channel

        return image_mean, mask

    return image, mask


def load_segmentation_data(data_dir, split="train", single_channel=False):
    """Create a tf.data.Dataset for the segmentation task."""
    task_dir = os.path.join(data_dir, "segmentation")

    assert split in ["train", "val"], "Split should be train or val."
    if split == "train":
        split_dir = os.path.join(task_dir, "train2017")
    elif split == "val":
        split_dir = os.path.join(task_dir, "val2017")

    # Create list of (filename, label) pairs
    files = [filename for filename in os.listdir(split_dir) if "image" in filename]
    images = [os.path.join(split_dir, filename) for filename in files]
    masks = [os.path.join(split_dir, filename.replace("image", "mask")) for filename in files]

    # Create a tf.data.Dataset from filenames and labels
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.map(
        lambda image, mask: tf.py_function(
            func=parse_function_segmentation,
            inp=[image, mask, single_channel],
            Tout=(tf.float32, tf.float32),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset


def create_dataset_segmentation(data_dir, split, batch_size, single_channel=False):
    """Load the data and prepare it as a batched tf.data.Dataset."""
    dataset = load_segmentation_data(data_dir, split, single_channel)
    return dataset.shuffle(500).batch(batch_size).prefetch(tf.data.AUTOTUNE)
