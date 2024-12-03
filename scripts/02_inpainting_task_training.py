import os
import sys
import argparse
import tensorflow as tf
import numpy as np

from abc import ABCMeta
from typing import Tuple, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.resnet import ResNet18
from src.weights.load_weights import load_model_weights
from src.weights.weights import weights_collection


def generate_coco_dataset_descriptor(dataset_path: str) -> dict:
    """
    Generate a dataset descriptor for COCO dataset.
    
    Args:
        dataset_path (str): COCO dataset folder path.
    
    Returns:
        dict: A dictionary containing the path of training and validation images.
    """
    descriptor = {}

    # Train and validation split
    train_path = os.path.join(dataset_path)
    train_samples = [
        os.path.join(train_path, f) for f in os.listdir(train_path)[:30000] if f.endswith(".jpg")
    ]
    total_samples = len(train_samples)
    train_size = int(0.8 * total_samples)
    indices = np.random.RandomState(42).permutation(total_samples)
    shuffled_samples = np.array(train_samples)[indices]
    descriptor["train"] = shuffled_samples[:train_size].tolist()
    descriptor["validation"] = shuffled_samples[train_size:].tolist()

    return descriptor


def create_mask(
    mask: tf.Tensor, input_shape: Tuple = (224, 224), mask_size: Tuple = (32, 32)
) -> tf.Tensor:
    """
    Create a random mask on the input tensor.
    
    Args:
        mask (tf.Tensor): Initial mask tensor.
        input_shape (Tuple): Shape of the input image.
        mask_size (Tuple): Size of the mask patches.
    
    Returns:
        tf.Tensor: Updated mask tensor with zeroed-out patches.
    """
    x_mask = tf.random.uniform((), 0, input_shape[0] - mask_size[0], dtype=tf.int32)
    y_mask = tf.random.uniform((), 0, input_shape[1] - mask_size[1], dtype=tf.int32)
    mask = tf.concat(
        [
            mask[:x_mask, :],
            tf.concat(
                [
                    mask[x_mask : x_mask + mask_size[0], :y_mask],
                    tf.fill([mask_size[0], mask_size[1]], 0),
                    mask[x_mask : x_mask + mask_size[0], y_mask + mask_size[1] :],
                ],
                axis=1,
            ),
            mask[x_mask + mask_size[0] :, :],
        ],
        axis=0,
    )
    return mask


class InpaintingDataLoader(metaclass=ABCMeta):
    """
    Data loader for inpainting task.
    """
    def __init__(
        self,
        epochs: int,
        batch_size: int,
        dataset_descriptor: dict,
        input_shape: Tuple = (224, 224),
        mask_size: Tuple = (32, 32),
        shuffle_buffer_size: int = 1000,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.dataset_descriptor = dataset_descriptor
        self.input_shape = input_shape
        self.mask_size = mask_size
        self.shuffle_buffer_size = shuffle_buffer_size

    def _read_decode_resize(self, images_path: str) -> tf.Tensor:
        """
        Read, decode, and resize an image.
        
        Args:
            images_path (str): Image path.
        
        Returns:
            tf.Tensor: Resized image tensor.
        """
        gt_image = tf.io.read_file(images_path)
        gt_image = tf.io.decode_jpeg(gt_image, channels=3)
        # resize image
        gt_image = tf.image.resize(gt_image, size=self.input_shape)

        return gt_image

    def _produce_mask(self, gt_image: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Generate a pseudo-label by applying five random masks to the input image.
        
        Args:
            gt_image (tf.Tensor): Ground truth image tensor.
        
        Returns:
            Tuple[tf.Tensor, tf.Tensor]: normalized masked image and ground truth image.
        """
        # CREATE MASK
        mask = tf.fill(self.input_shape, 255)
        # Patch 1
        mask = create_mask(mask, self.input_shape, self.mask_size)
        # Patch 2
        mask = create_mask(mask, self.input_shape, self.mask_size)
        # Patch 3
        mask = create_mask(mask, self.input_shape, self.mask_size)
        # Patch 4
        mask = create_mask(mask, self.input_shape, self.mask_size)
        # Patch 5
        mask = create_mask(mask, self.input_shape, self.mask_size)

        # Generate masked image
        mask = tf.cast(mask, dtype=tf.float32)
        mask = tf.repeat(tf.expand_dims(mask, axis=-1), repeats=3, axis=-1)
        masked_image = gt_image * (mask / 255.0)

        # Normalize masked image and gt image
        masked_image = tf.cast(masked_image, tf.float32) / 255.0
        gt_image = tf.cast(gt_image, tf.float32) / 255.0

        return masked_image, gt_image

    def get_data_loader(self, subset_name: str) -> tf.data.Dataset:
        """
        Create a data loader for a specified set.
        
        Args:
            subset_name (str): 'train' or 'validation'.
        
        Returns:
            tf.data.Dataset: Dataset object with applied preprocessing.
        """
        if subset_name == "train":
            image_paths = self.dataset_descriptor["train"]
        else:
            image_paths = self.dataset_descriptor["validation"]

        dataset = tf.data.Dataset.from_tensor_slices(image_paths)

        dataset = dataset.shuffle(
            buffer_size=self.shuffle_buffer_size, reshuffle_each_iteration=True
        )

        dataset = dataset.map(self._read_decode_resize, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.map(self._produce_mask, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(self.batch_size, drop_remainder=True)

        return dataset.prefetch(tf.data.AUTOTUNE)


class PerceptualLoss(tf.keras.losses.Loss):
    """
    Custom perceptual loss using a pre-trained VGG19 model.
    """
    def __init__(self):
        super().__init__()
        vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
        self.layer_names = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in self.layer_names]
        self.vgg_model = tf.keras.Model([vgg.input], outputs)

    def call(self, y_true, y_pred):
        """
        Compute the perceptual loss.
        
        Args:
            y_true (tf.Tensor): Ground truth image tensor.
            y_pred (tf.Tensor): Predicted image tensor.
        
        Returns:
            tf.Tensor: Perceptual loss value.
        """
        y_true_vgg = tf.keras.applications.vgg19.preprocess_input(y_true * 255.0)
        y_pred_vgg = tf.keras.applications.vgg19.preprocess_input(y_pred * 255.0)

        y_true_features = self.vgg_model(y_true_vgg)
        y_pred_features = self.vgg_model(y_pred_vgg)

        loss = 0.0
        for y_true_feat, y_pred_feat in zip(y_true_features, y_pred_features):
            loss += tf.reduce_mean(tf.square((y_true_feat / 255.0) - (y_pred_feat / 255.0)))

        return loss / len(self.layer_names)


def get_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--epochs", default=100, type=int, help="Number of epochs to train the model."
    )
    parser.add_argument("-bs", "--batch_size", default=8, type=int, help="Batch size.")
    parser.add_argument("-lr", "--learning_rate", default=1e-3, type=float, help="Learning rate.")
    parser.add_argument(
        "-t", "--train_dir", default="./data/train2017", help="Train set folder path."
    )
    parser.add_argument(
        "-s", "--save_weights_dir", default="models/inpainting_ckpt", help="Folder to save weights."
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments for training configuration.
    args = get_args()
    
    # Create the directory to save model weights if it does not exist.
    os.makedirs(args.save_weights_dir, exist_ok=False)

    # Generate a descriptor for the dataset, splitting it into training and validation subsets.
    dataset_descriptor = generate_coco_dataset_descriptor(args.train_dir)

    # Initialize the data loader for the inpainting task with specified parameters.
    inpaint_dataloader = InpaintingDataLoader(args.epochs, args.batch_size, dataset_descriptor)

    # Retrieve the data loader for the training and validation subsets.
    train_dataloader = inpaint_dataloader.get_data_loader("train")
    validation_dataloader = inpaint_dataloader.get_data_loader("validation")
    
    # Instantiate the custom perceptual loss function using a pre-trained VGG19 model.
    loss_fn = PerceptualLoss()
    
    # Initialize the ResNet18 model configured for the inpainting task.
    model = ResNet18(mode="inpainting")
    
    # Load pre-trained weights for the ResNet18 no-top model from a predefined collection.
    model = load_model_weights(weights_collection, model, 1000, False, "resnet18")

    model.summary()

    # Define a list of callbacks to be used during training for monitoring and optimizing the process.
    _callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            args.save_weights_dir + "weights-{epoch:03d}-{val_loss:.4f}.weights.h5",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
            mode="min",
            save_weights_only=True,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=args.save_weights_dir),
    ]
    callbacks = tf.keras.callbacks.CallbackList(_callbacks, add_history=True, model=model)

    # Compile the model, specifying the optimizer, custom loss function, and evaluation metrics.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=loss_fn,
        metrics=["mae", "mse"],
    )
    
    # Train the model using the training and validation data loaders, applying the specified callbacks.
    model.fit(
        train_dataloader,
        validation_data=validation_dataloader,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )
