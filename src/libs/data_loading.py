import os
import json
import numpy as np
import tensorflow as tf

# <------- Classification data loader functions ------->

pascal_voc_classes = [
    "person", "bird", "cat", "cow", "dog", "horse", "sheep",
    "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
    "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor",
]

def objects_to_labels(objects, classes=None):
    """
    Converts a list of objects into a one-hot encoded label vector based on the provided classes.

    Args:
        objects (list): List of object names (strings) from an annotation file.
        classes (list): List of class names to consider (defaults to PASCAL VOC classes).

    Returns:
        np.ndarray: One-hot encoded vector of shape [num_classes], where 1 indicates the presence of a class.
    """  
    if not classes:
        classes = pascal_voc_classes

    # Assert all classes of interest are valid pascal voc classes
    assert all(class_name in pascal_voc_classes for class_name in classes)

    # Create dict of label2int just with classes of interest
    label2int = {label: index for index, label in enumerate(classes)}

    # Create one-hot encoded vector
    labels = np.zeros(len(classes))
    for obj in objects:
        if obj in classes:
            labels[label2int[obj]] = 1
    return labels


def parse_function_classification(filename, label, data_dir, single_channel=False):
    """
    Loads and processes an image for classification.

    Args:
        filename (tf.Tensor): Tensor containing the filename of the image.
        label (tf.Tensor): Tensor containing the label associated with the image.
        data_dir (tf.Tensor): Tensor containing the base directory of the dataset.
        single_channel (bool): Whether to convert the image to a single grayscale channel.

    Returns:
        tuple: Processed image and its associated label.
    """
    # Load the image from file
    filename = filename.numpy().decode("utf-8")
    data_dir = data_dir.numpy().decode("utf-8")
    image = np.load(os.path.join(data_dir, "classification", f"{filename}.npy"))

    if single_channel:
        # Average the three channels using luminosity method
        image_mean = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        # Expand dimensions to make it (height, width, 1)
        image_mean = np.expand_dims(image_mean, axis=-1)
        
        # Ensure shape is (224, 224, 1)
        image_mean = image_mean.astype(np.float32)
        return image_mean, label

    # Ensure shape is (224, 224, 3)
    image = image.astype(np.float32)
    return image, label


def load_classification_data(data_dir, split_list_file, single_channel=False, classes=None):
    """
    Loads and prepares data for the classification task.

    Args:
        data_dir (str): Base directory of the dataset.
        split_list_file (str): Path to the file containing image filenames for a specific split (train or val).
        single_channel (bool): Whether to convert images to single-channel grayscale.
        classes (list): List of class names to consider (defaults to PASCAL VOC classes).

    Returns:
        tf.data.Dataset: Dataset containing pairs of processed images and one-hot encoded labels.
    """

    task_dir = os.path.join(data_dir, "classification")

    # Read data.json file containing image names and corresponding objects
    with open(os.path.join(task_dir, "data.json"), "r") as file:
        annotations = json.load(file)

    # Read train.txt or val.txt file containing list of images for each split
    with open(split_list_file) as file:
        split_files = [line.rstrip() for line in file]

    # Create list of filenames and labels
    filenames = []
    labels = []
    for filename in split_files:
        filenames.append(filename)
        labels.append(objects_to_labels(annotations[filename], classes))

    labels = np.array(labels).astype(np.float32)

    # Create a tf.data.Dataset from filenames and labels pairs
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    
    # Map function with explicit shape inference
    def map_func(filename, label):
        image, label = tf.py_function(
            func=parse_function_classification,
            inp=[filename, label, data_dir, single_channel],
            Tout=(tf.float32, tf.float32),
        )
        
        # Set explicit shapes based on single_channel flag
        if single_channel:
            image.set_shape([224, 224, 1])
        else:
            image.set_shape([224, 224, 3])
        
        return image, label

    dataset = dataset.map(
        map_func,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset


def create_dataset_classification(data_dir, split_list_file, batch_size, single_channel=False, classes=None):
    """
    Creates a batched and prefetched dataset for classification.

    Args:
        data_dir (str): Base directory of the dataset.
        split_list_file (str): Path to the file containing image filenames for a specific split.
        batch_size (int): Number of samples per batch.
        single_channel (bool): Whether to convert images to single-channel grayscale.
        classes (list): List of class names to consider (defaults to PASCAL VOC classes).

    Returns:
        tf.data.Dataset: Batched and prefetched dataset for classification.
    """
    dataset = load_classification_data(data_dir, split_list_file, single_channel, classes)
    
    # Batch with known shape
    if single_channel:
        return dataset.shuffle(500).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    else:
        return dataset.shuffle(500).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


# <------- Segmentation data loader functions ------->


def parse_function_segmentation(image_path, mask_path, single_channel):
    """
    Loads and processes an image and its corresponding mask for segmentation.

    Args:
        image_path (tf.Tensor): Tensor containing the file path of the input image.
        mask_path (tf.Tensor): Tensor containing the file path of the corresponding mask.
        single_channel (bool): Whether to convert the image to single-channel grayscale.

    Returns:
        tuple: Processed image and its corresponding mask.
    """

    # Load the image from file
    image_path = image_path.numpy().decode("utf-8")
    mask_path = mask_path.numpy().decode("utf-8")

    image = np.load(image_path)
    mask = np.load(mask_path)

    if single_channel:
        # Expand dimensions of image to make it (height, width, 1)
        image = np.expand_dims(image[..., :1], axis=-1)
    
    # Ensure correct dtype
    image = image.astype(np.float32)
    mask = mask.astype(np.float32)

    return image, mask


def load_segmentation_data(data_dir, split="train", single_channel=False):
    """
    Loads and prepares data for the segmentation task.

    Args:
        data_dir (str): Base directory of the dataset.
        split (str): Dataset split to use ('train', 'val', or 'test').
        single_channel (bool): Whether to convert images to single-channel grayscale.

    Returns:
        tf.data.Dataset: Dataset containing pairs of processed images and masks.
    """

    task_dir = os.path.join(data_dir, "segmentation")

    assert split in ["train", "val", "test"], "Split should be train, val or test."
    if split == "train" or split == "val":
        split_dir = os.path.join(task_dir, "train2017")
    elif split == "test":
        split_dir = os.path.join(task_dir, "val2017")

    # Create lists of images and masks
    file_search_substr = "image"
    if single_channel:
        file_search_substr = "inputgray"
    files = [filename for filename in os.listdir(split_dir) if file_search_substr in filename]
    images = [os.path.join(split_dir, filename) for filename in files]
    masks = [
        os.path.join(split_dir, filename.replace(file_search_substr, "mask")) for filename in files
    ]

    # For train and val, split the original "train" split
    split_idx = int(0.8 * len(images))
    if split == "train":
        images = images[:split_idx]
        masks = masks[:split_idx]
    elif split == "val":
        images = images[split_idx:]
        masks = masks[split_idx:]

    # Create a tf.data.Dataset from images and masks
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    
    # Map function with explicit shape inference
    def map_func(image, mask):
        image, mask = tf.py_function(
            func=parse_function_segmentation,
            inp=[image, mask, single_channel],
            Tout=(tf.float32, tf.float32),
        )
        
        # Set explicit shapes 
        if single_channel:
            # Single channel image is 224x224x1
            image.set_shape([224, 224, 1])
        else:
            # Color image is 224x224x3 or 224x224x4
            image.set_shape([224, 224, None])
        
        # Segmentation mask
        mask.set_shape([224, 224, 4])
        
        return image, mask

    dataset = dataset.map(
        map_func,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset


def create_dataset_segmentation(data_dir, split, batch_size, single_channel=False):
    """
    Creates a batched and prefetched dataset for segmentation.

    Args:
        data_dir (str): Base directory of the dataset.
        split (str): Dataset split to use ('train', 'val', or 'test').
        batch_size (int): Number of samples per batch.
        single_channel (bool): Whether to convert images to single-channel grayscale.

    Returns:
        tf.data.Dataset: Batched and prefetched dataset for segmentation.
    """
    dataset = load_segmentation_data(data_dir, split, single_channel)
    return dataset.shuffle(500).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)