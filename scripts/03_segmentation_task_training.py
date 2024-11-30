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
        )  # Sum over spatial dimensions
        union = (
            tf.reduce_sum(y_true_class, axis=(1, 2))
            + tf.reduce_sum(y_pred_class, axis=(1, 2))
            - intersection
        )

        # Avoid division by zero by using a conditional operation
        iou = tf.where(union > 0, intersection / union, tf.ones_like(union))
        ious.append(iou)

    # Compute mean IoU over all classes
    mean_iou = tf.reduce_mean(tf.stack(ious, axis=0), axis=0)  # Average over classes

    return tf.reduce_mean(mean_iou)  # Average over batch


class TrainingProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback to track and save training progress with improved model saving."""

    def __init__(self, checkpoint_dir="models", save_freq=1, save_format="h5"):
        """
        Initialize the callback with configurable save format.

        Args:
            checkpoint_dir (str): Directory to save checkpoints
            save_freq (int): Frequency of saving checkpoints (in epochs)
            save_format (str): Format to save the model ('h5' or 'tf')
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
        """Load training history from file if it exists."""
        history_path = os.path.join(self.checkpoint_dir, "training_history.npy")
        if os.path.exists(history_path):
            try:
                return np.load(history_path, allow_pickle=True).item()
            except:
                return self._initialize_history()
        return self._initialize_history()

    def _load_best_val_loss(self):
        """Load best validation loss from file if it exists."""
        best_loss_path = os.path.join(self.checkpoint_dir, "best_val_loss.npy")
        if os.path.exists(best_loss_path):
            return float(np.load(best_loss_path))
        return float("inf")

    def _initialize_history(self):
        """Initialize empty history dictionary."""
        return {"loss": [], "val_loss": [], "iou_metric": [], "val_iou_metric": [], "lr": []}

    def _save_model(self, filepath):
        """Save model with proper configuration."""
        if self.save_format == "h5":
            # Save as HDF5 format
            self.model.save(filepath, save_format="h5")
        else:
            # Save in TensorFlow SavedModel format
            self.model.save(filepath, save_format="tf")

    def on_epoch_end(self, epoch, logs=None):
        """Handle end of epoch operations including model saving."""
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

    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    # set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
    class_weights = np.array([1.0, 1.0, 1.0, 0.05])
    dice_loss = sm.losses.DiceLoss(class_weights=class_weights)
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=total_loss,
        # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[iou_metric],
    )

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
    class_weights = np.array([1, 1, 1, 0.05])
    dice_loss = sm.losses.DiceLoss(class_weights=class_weights)
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    return total_loss


# Modified feature monitoring to handle dynamic shapes
class FeatureMonitorCallback(tf.keras.callbacks.Callback):
    def __init__(self, reference_model, monitor_layers):
        super().__init__()
        self.reference_model = reference_model
        self.monitor_layers = monitor_layers
        self.feature_distances = {layer: [] for layer in monitor_layers}

    def on_epoch_end(self, epoch, logs=None):
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
    """Train with gradual unfreezing strategy, optimized for ResNet + skip connections."""

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
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    args = get_args()
    data_dir = os.path.join(args.data_path, "processed", "coco")

    # Create checkpoint directory if not exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.single_channel:
        model = ResNet18((224, 224, 1), mode="segmentation")
    else:
        model = ResNet18((224, 224, 3), mode="segmentation")
    print(model.summary())

    if args.pretrained_model:
        print("Loading model weights...")
        load_encoder_weights(model, args.pretrained_model)

    # Load data and create dataset
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
