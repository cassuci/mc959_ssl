import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import re
import argparse

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.resnet import ResNet18, load_encoder_weights
from src.libs.data_loading import create_dataset_segmentation


def iou_metric(
    y_true: tf.Tensor, y_pred: tf.Tensor, num_classes: int = 3, threshold: float = 0.5
) -> tf.Tensor:
    """
    Calculates the Intersection over Union (IoU) metric for multi-class segmentation.

    Args:
        y_true (tf.Tensor): Ground truth tensor with shape [batch, height, width, num_classes].
        y_pred (tf.Tensor): Predicted tensor with shape [batch, height, width, num_classes].
        num_classes (int): Number of classes in the segmentation task.
        threshold (float): Threshold for binarizing predictions.

    Returns:
        tf.Tensor: Mean IoU across all classes and batches.
    """

    # Binarize predictions
    y_pred_bin = tf.cast(y_pred > threshold, tf.float32)

    # Initialize list to store IoU for each class
    ious = []

    for class_idx in range(num_classes):
        # Extract predictions and ground truth for the current class
        y_true_class = y_true[..., class_idx]
        y_pred_class = y_pred_bin[..., class_idx]

        # Compute intersection and union
        intersection = tf.reduce_sum(
            y_true_class * y_pred_class, axis=(1, 2)
        )
        # Sum over spatial dimensions
        union = (
            tf.reduce_sum(y_true_class, axis=(1, 2))
            + tf.reduce_sum(y_pred_class, axis=(1, 2))
            - intersection
        )

        # Avoid division by zero by using a conditional operation
        iou = tf.where(union > 0, intersection / union, tf.ones_like(union))
        ious.append(iou)

    # Compute mean IoU over all classes
    mean_iou = tf.reduce_mean(tf.stack(ious, axis=0), axis=0)

    # Average over batch
    return tf.reduce_mean(mean_iou)


class TrainingProgressCallback(tf.keras.callbacks.Callback):

    def __init__(self, checkpoint_dir="models", save_freq=1, save_format="h5"):
        """
        Custom callback to track training progress, save periodic checkpoints, and save the best model.

        Args:
            checkpoint_dir (str): Directory to save model checkpoints.
            save_freq (int): Frequency of saving checkpoints (in epochs).
            save_format (str): Format for saving the model ('h5' or 'tf').

        Attributes:
            checkpoint_dir (str): Path to save the checkpoints.
            save_freq (int): Frequency of checkpoint saving.
            save_format (str): Format for saving models ('h5' or 'tf').
            best_val_loss (float): Best validation loss encountered during training.
            history (dict): Tracks training and validation metrics.
        """
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.save_freq = save_freq
        self.save_format = save_format
        self.best_val_loss = float("inf")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Load existing history if it exists
        self.history = self._load_history()
        self.best_val_loss = self._load_best_val_loss()


    def _load_history(self):
        """
        Loads the training history from a file if it exists, otherwise initializes an empty history.

        Returns:
            dict: Training history containing loss, metrics, and learning rate logs.
        """
        history_path = os.path.join(self.checkpoint_dir, "training_history.npy")
        if os.path.exists(history_path):
            try:
                return np.load(history_path, allow_pickle=True).item()
            except:
                return self._initialize_history()
        return self._initialize_history()


    def _load_best_val_loss(self):
        """
        Loads the best validation loss from a file if it exists, otherwise initializes it to infinity.

        Returns:
            float: The best validation loss encountered so far.
        """
        best_loss_path = os.path.join(self.checkpoint_dir, "best_val_loss.npy")
        if os.path.exists(best_loss_path):
            return float(np.load(best_loss_path))
        return float("inf")


    def _initialize_history(self):
        """
        Initializes an empty dictionary to track loss, validation loss, IoU metrics, and learning rate.

        Returns:
            dict: Empty training history dictionary with predefined keys.
        """
        return {"loss": [], "val_loss": [], "iou_metric": [], "val_iou_metric": [], "lr": []}


    def _save_model(self, filepath):
        """
        Saves the model in the specified format (HDF5 or TensorFlow SavedModel).

        Args:
            filepath (str): Path where the model will be saved.
        """
        if self.save_format == "h5":
            # Save as HDF5 format
            self.model.save(filepath, save_format="h5")
        else:
            # Save in TensorFlow SavedModel format
            self.model.save(filepath, save_format="tf")


    def on_epoch_end(self, epoch, logs=None):
        """
        Handles operations at the end of each training epoch, including:
            - Updating training history.
            - Saving periodic model checkpoints.
            - Saving the best model based on validation loss.

        Args:
            epoch (int): Current epoch number.
            logs (dict): Dictionary containing training and validation metrics.
        """
        logs = logs or {}

        # Update history
        for metric in self.history.keys():
            if metric in logs:
                self.history[metric].append(logs[metric])
            elif metric == "lr":
                lr = K.get_value(self.model.optimizer.learning_rate)
                self.history["lr"].append(lr)

        # Save periodic checkpoint
        if (epoch + 1) % self.save_freq == 0:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"segmentation_model_epoch_{epoch + 1:03d}"
            )
            if self.save_format == "h5":
                checkpoint_path += ".h5"
            self._save_model(checkpoint_path)
            print(f"\nSaved periodic checkpoint for epoch {epoch + 1}")

        # Save best model
        current_val_loss = logs.get("val_loss", float("inf"))
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            best_model_path = os.path.join(self.checkpoint_dir, "best_segmentation_model")
            if self.save_format == "h5":
                best_model_path += ".h5"

            # Save complete model
            self._save_model(best_model_path)

            # Save best validation loss
            np.save(os.path.join(self.checkpoint_dir, "best_val_loss.npy"), self.best_val_loss)
            print(f"\nNew best model saved with validation loss: {self.best_val_loss:.6f}")

        # Save training history
        history_path = os.path.join(self.checkpoint_dir, "training_history.npy")
        np.save(history_path, self.history)


def train_model(
    model,
    train_dataset,
    val_dataset,
    epochs=10,
    initial_epoch=0,
    checkpoint_dir="models",
    load_latest_checkpoint=True,
):
    """
    Trains the segmentation model with specified datasets, loss function, and callbacks.

    Args:
        model (tf.keras.Model): Segmentation model to train.
        train_dataset (tf.data.Dataset): Dataset for training.
        val_dataset (tf.data.Dataset): Dataset for validation.
        epochs (int): Total number of epochs to train.
        initial_epoch (int): Epoch to start training from (useful for checkpoint resumption).
        checkpoint_dir (str): Directory for saving checkpoints and history.
        load_latest_checkpoint (bool): Whether to load the latest checkpoint before training.

    Returns:
        tuple: Trained model and training history.
    """

    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create a TensorFlow checkpoint
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpoint_dir, max_to_keep=5
    )

    # Determine if we should load the latest checkpoint
    if load_latest_checkpoint:
        latest_checkpoint = checkpoint_manager.latest_checkpoint
        if latest_checkpoint:
            try:
                # Restore the checkpoint
                checkpoint.restore(latest_checkpoint).expect_partial()

                # Extract epoch number from checkpoint filename
                epoch_match = re.search(r"ckpt-(\d+)", latest_checkpoint)
                if epoch_match:
                    initial_epoch = int(epoch_match.group(1))

                print(f"Loaded checkpoint from epoch {initial_epoch}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting training from scratch...")
                initial_epoch = 0

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=get_segmentation_loss(),
        metrics=[iou_metric],
    )

    # Define training callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0, patience=5, verbose=1, mode="auto"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1, mode="auto"
        ),
        TrainingProgressCallback(checkpoint_dir=checkpoint_dir, save_format='tf')
    ]

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        initial_epoch=initial_epoch,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
    )

    return model, history


def get_segmentation_loss():
    """
    Creates and returns the total segmentation loss function by combining Dice loss and Focal loss.

    Returns:
        loss: Combined Dice and Focal loss function for segmentation tasks.
    """
    class_weights = np.array([1, 1, 1, 0.05])
    dice_loss = sm.losses.DiceLoss(class_weights=class_weights)
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    return total_loss


class FeatureMonitorCallback(tf.keras.callbacks.Callback):
    def __init__(self, reference_model, monitor_layers):
        """
        Custom callback to monitor feature extraction layers and detect feature drift during training.

        Args:
            reference_model (tf.keras.Model): Model used as a reference for comparison.
            monitor_layers (list): List of layer names to monitor for feature drift.

        Attributes:
            reference_model (tf.keras.Model): Reference model for feature comparison.
            monitor_layers (list): Layers to monitor during training.
            feature_distances (dict): Tracks cosine similarity distance for monitored layers.
        """
        super().__init__()
        self.reference_model = reference_model
        self.monitor_layers = monitor_layers
        self.feature_distances = {layer: [] for layer in monitor_layers}

    def on_epoch_end(self, epoch, logs=None):
        """
        Monitors feature drift at the end of each epoch by comparing features from specific layers 
        of the current model and the reference model.

        Steps:
            - Extracts a batch of data from the validation dataset.
            - Compares the cosine similarity between features of monitored layers.
            - Records the similarity distances and warns if drift exceeds a threshold.

        Args:
            epoch (int): Current epoch number.
            logs (dict): Training and validation metrics for the epoch (optional).
        """

        try:
            # Get a batch of data
            for x_batch, _ in val_dataset.take(1):
                # Create feature extraction models
                for layer_name in self.monitor_layers:
                    if (
                        layer_name in self.reference_model.layers
                        and layer_name in self.model.layers
                    ):
                        ref_layer = self.reference_model.get_layer(layer_name)
                        curr_layer = self.model.get_layer(layer_name)

                        # Get features
                        ref_features = ref_layer(x_batch)
                        curr_features = curr_layer(x_batch)

                        # Calculate distance
                        if hasattr(ref_features, "shape") and hasattr(curr_features, "shape"):
                            distance = tf.reduce_mean(
                                tf.keras.losses.cosine_similarity(
                                    tf.reshape(ref_features, [tf.shape(ref_features)[0], -1]),
                                    tf.reshape(curr_features, [tf.shape(curr_features)[0], -1]),
                                )
                            )
                            self.feature_distances[layer_name].append(float(distance))

                            if float(distance) > 0.5:
                                print(
                                    f"\nWarning: Large feature drift in {layer_name}: {distance:.3f}"
                                )
        except Exception as e:
            print(f"Feature monitoring warning: {str(e)}")


def train_gradual_unfreeze(model, train_dataset, val_dataset, epochs=30, checkpoint_dir="models"):
    """
    Trains the segmentation model using a gradual unfreezing strategy for better fine-tuning.

    Args:
        model (tf.keras.Model): Segmentation model to train.
        train_dataset (tf.data.Dataset): Dataset for training.
        val_dataset (tf.data.Dataset): Dataset for validation.
        epochs (int): Total number of epochs for training across all phases.
        checkpoint_dir (str): Directory for saving model checkpoints.

    Returns:
        tuple: Trained model and combined training history across all phases.
    """
    # Create callbacks that will be used in all phases
    base_callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",  # Monitor validation loss
            factor=0.5,  # Reduce LR by half
            patience=3,  # Wait 3 epochs before reducing LR
            min_lr=1e-6,  # Set a minimum learning rate
            verbose=1,  # Print messages when LR is updated
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        TrainingProgressCallback(checkpoint_dir=checkpoint_dir, save_format="tf"),
    ]

    # Store original model weights for feature monitoring
    reference_model = tf.keras.models.clone_model(model)
    reference_model.set_weights(model.get_weights())

    # Add feature monitoring for important skip connection layers
    feature_monitor = FeatureMonitorCallback(
        reference_model=reference_model,
        # monitor_layers=['conv2_block3_out', "conv3_block4_out", "conv4_block6_out"]  # Key layers with skip connections
        monitor_layers=["block_1_0", "block_2_1", "block_3_0"],  # Key layers with skip connections
    )

    # Phase 1: Train only decoder (3-5 epochs)
    print("\nPhase 1: Training decoder only...")
    for layer in model.layers:
        if not "decoder" in layer.name:
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=get_segmentation_loss(),
        metrics=[iou_metric],
    )

    phase1_epochs = 5
    history1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=phase1_epochs,
        callbacks=base_callbacks,
        verbose=1,
    )

    # Phase 2: Unfreeze later blocks (where skip connections come from)
    print("\nPhase 2: Fine-tuning later encoder blocks...")
    for layer in model.layers:
        if "block_3_1" in layer.name or "block_2_0" in layer.name:
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=get_segmentation_loss(),
        metrics=[iou_metric],
    )

    phase2_epochs = 10
    history2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        initial_epoch=phase1_epochs,
        epochs=phase1_epochs + phase2_epochs,
        callbacks=base_callbacks + [feature_monitor],  # Add feature monitoring in phase 2
        verbose=1,
    )

    # Phase 3: Unfreeze remaining blocks with very low learning rate
    print("\nPhase 3: Fine-tuning entire network...")
    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=get_segmentation_loss(),
        metrics=[iou_metric],
    )

    history3 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        initial_epoch=phase1_epochs + phase2_epochs,
        epochs=epochs,
        callbacks=base_callbacks + [feature_monitor],
        verbose=1,
    )

    # Combine histories
    combined_history = {}
    for key in history1.history:
        combined_history[key] = (
            history1.history[key] + history2.history[key] + history3.history[key]
        )

    return model, combined_history


def get_args():
    """
    Parses command-line arguments for the segmentation training script.

    Returns:
        argparse.Namespace: Parsed arguments including paths and training settings.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data", type=str, help="Dataset folder path")
    parser.add_argument(
        "--two_phases_train", action="store_true", help="Allow training in two phases"
    )
    parser.add_argument(
        "--pretrained_model", default=None, type=str, help="Path to pretrained model"
    )
    parser.add_argument("--single_channel", action="store_true", help="To use grayscale images")
    parser.add_argument(
        "--checkpoint_dir",
        default="models/segmentation_ckpt",
        type=str,
        help="Prefix to the saved model path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    """
    Main function for training a segmentation model.
    
    Steps:
        1. Parse command-line arguments.
        2. Initialize the model.
        3. Load datasets for training and validation.
        4. Train the model with specified settings.
        5. Save the trained model.
    """

    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    args = get_args()
    data_dir = os.path.join(args.data_path, "processed", "coco")

    # Create checkpoint directory if not exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize the model
    if args.single_channel:
        model = ResNet18((224, 224, 1), mode="segmentation")
    else:
        model = ResNet18((224, 224, 3), mode="segmentation")
    print(model.summary())

    # Load pretrained weights if specified
    if args.pretrained_model:
        print("Loading model weights...")
        load_encoder_weights(model, args.pretrained_model)

    # Create train and val datasets
    print("Loading data and creating dataset...")
    train_dataset = create_dataset_segmentation(
        data_dir,
        split="train",
        batch_size=32,
        single_channel=args.single_channel,
    )
    val_dataset = create_dataset_segmentation(
        data_dir,
        split="val",
        batch_size=32,
        single_channel=args.single_channel,
    )

    # Train the model
    print("Training the model...")
    if args.two_phases_train:
        trained_model, history = train_gradual_unfreeze(
            model,
            train_dataset,
            val_dataset,
            checkpoint_dir=args.checkpoint_dir,
            epochs=30,
        )
    else:
        trained_model, history = train_model(
            model, train_dataset, val_dataset, checkpoint_dir=args.checkpoint_dir, epochs=20
        )

    # Save the final model in both formats
    # TensorFlow Checkpoint
    final_tf_checkpoint_path = os.path.join(args.checkpoint_dir, "final_model")
    checkpoint = tf.train.Checkpoint(model=trained_model)
    checkpoint.save(final_tf_checkpoint_path)

    # H5 Weights
    final_h5_path = os.path.join(args.checkpoint_dir, "final_segmentation_model.weights.h5")
    trained_model.save_weights(final_h5_path)

    print(f"Final model saved as TF checkpoint to {final_tf_checkpoint_path}")
    print(f"Final model weights saved to {final_h5_path}")

"""
python scripts/03_segmentation_task_training.py --two_phases_train --single_channel --checkpoint_dir models/checkpoints_seg_resnet18_new_decoder_vgg --pretrained_model models/checkpoints_color_resnet18_new_decoder_vgg
"""
