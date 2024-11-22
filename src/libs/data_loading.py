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
    """Load images and masks with explicit type casting and shape checking."""
    # Load the image from file
    image_path = image_path.numpy().decode("utf-8")
    mask_path = mask_path.numpy().decode("utf-8")

    image = np.load(image_path).astype(np.float32)
    mask = np.load(mask_path).astype(np.float32)
    
    # Normalize image to [0,1] range if not already
    if image.max() > 1.0:
        image = image / 255.0
    
    # Ensure mask is one-hot encoded
    if len(mask.shape) == 2:  # If mask is (H,W) format
        # Convert to one-hot, assuming 4 classes (0,1,2,3)
        mask = tf.one_hot(mask.astype(np.int32), depth=4)
    
    if single_channel:
        # Convert RGB to grayscale using proper weights
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        image = np.expand_dims(image, axis=-1)

    # Ensure proper shapes
    if len(image.shape) != 3:
        raise ValueError(f"Image shape should be (H,W,C), got {image.shape}")
    if len(mask.shape) != 3:
        raise ValueError(f"Mask shape should be (H,W,C), got {mask.shape}")
        
    return image, mask


def load_segmentation_data(data_dir, split="train", single_channel=False):
    """Create a tf.data.Dataset for the segmentation task."""
    task_dir = os.path.join(data_dir, "segmentation")

    assert split in ["train", "val", "test"], "Split should be train, val or test."
    if split == "train" or split == "val":
        split_dir = os.path.join(task_dir, "train2017")
    elif split == "test":
        split_dir = os.path.join(task_dir, "val2017")

    # Create list of (filename, label) pairs
    file_search_substr = "image"
    if single_channel:
        file_search_substr = "inputgray"
    files = [filename for filename in os.listdir(split_dir) if file_search_substr in filename]
    images = [os.path.join(split_dir, filename) for filename in files]
    masks = [
        os.path.join(split_dir, filename.replace(file_search_substr, "mask")) for filename in files
    ]

    split_idx = int(0.8 * len(images))
    if split == "train":
        images = images[:split_idx]
        masks = masks[:split_idx]
        print(images[:2])
    elif split == "val":
        images = images[split_idx:]
        masks = masks[split_idx:]
    elif split == 'test':
        print(images[:2])

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
    """Load the data and prepare it as a batched tf.data.Dataset with explicit shapes."""
    dataset = load_segmentation_data(data_dir, split, single_channel)
    
    # Define output shapes and types
    if single_channel:
        output_shapes = ((None, None, 1), (None, None, 4))  # (H,W,1) for image, (H,W,4) for mask
    else:
        output_shapes = ((None, None, 3), (None, None, 4))  # (H,W,3) for image, (H,W,4) for mask
    
    output_types = (tf.float32, tf.float32)
    
    # Set shapes and types
    dataset = dataset.map(
        lambda x, y: (
            tf.ensure_shape(x, output_shapes[0]),
            tf.ensure_shape(y, output_shapes[1])
        )
    )
    
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)