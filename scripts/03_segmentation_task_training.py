import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import re
import argparse

os.environ['SM_FRAMEWORK'] = 'tf.keras'
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
        intersection = tf.reduce_sum(y_true_class * y_pred_class, axis=(1, 2))  # Sum over spatial dimensions
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
    """Custom callback to track and save training progress."""

    def __init__(self, checkpoint_dir="models", save_freq=1):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.save_freq = save_freq
        self.best_val_loss = float("inf")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Load existing history if it exists
        history_path = os.path.join(checkpoint_dir, "training_history.npy")
        if os.path.exists(history_path):
            try:
                self.history = np.load(history_path, allow_pickle=True).item()
            except:
                self.history = self._initialize_history()
        else:
            self.history = self._initialize_history()

        # Load best validation loss if it exists
        best_loss_path = os.path.join(checkpoint_dir, "best_val_loss.npy")
        if os.path.exists(best_loss_path):
            self.best_val_loss = float(np.load(best_loss_path))

    def _initialize_history(self):
        return {"loss": [], "val_loss": [], "iou_metric": [], "val_iou_metric": [], "lr": []}

    def on_epoch_end(self, epoch, logs=None):
        # Update history with available logs
        logs = logs or {}
        for metric in self.history.keys():
            if metric in logs:
                self.history[metric].append(logs[metric])
            elif metric == "lr":
                # Manually get current learning rate
                lr = K.get_value(self.model.optimizer.learning_rate)
                self.history["lr"].append(lr)

        # Save periodic checkpoint
        if (epoch + 1) % self.save_freq == 0:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"segmentation_model_epoch_{epoch + 1:03d}.h5"
            )
            self.model.save_weights(checkpoint_path)
            print(f"\nSaved periodic checkpoint for epoch {epoch + 1}")

        # Save best model
        if logs.get("val_loss", float("inf")) < self.best_val_loss:
            self.best_val_loss = logs["val_loss"]
            best_model_path = os.path.join(self.checkpoint_dir, "best_segmentation_model.h5")
            self.model.save_weights(best_model_path)
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
    class_weights = np.array([1., 1., 1., 0.1])
    dice_loss = sm.losses.DiceLoss(class_weights=class_weights) 
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=total_loss,
        #loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[iou_metric],
    )

    # Create callbacks for training
    class CustomCheckpointCallback(tf.keras.callbacks.Callback):
        def __init__(self, checkpoint_dir):
            super().__init__()
            self.checkpoint_dir = checkpoint_dir

        def on_epoch_end(self, epoch, logs=None):
            # Save TensorFlow checkpoint
            checkpoint_manager.save()

            # Save H5 checkpoint
            h5_path = os.path.join(
                self.checkpoint_dir, f"segmentation_model_epoch_{epoch + 1:03d}.h5"
            )
            self.model.save_weights(h5_path)
            print(f"\nSaved H5 checkpoint for epoch {epoch + 1}")

    callbacks = [
        CustomCheckpointCallback(checkpoint_dir),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0, patience=5, verbose=1, mode="auto"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1, mode="auto"
        ),
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


def get_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint file and extract its epoch number."""
    # First, look for H5 checkpoints
    import glob
    import re

    h5_files = sorted(
        glob.glob(os.path.join(checkpoint_dir, "segmentation_model_epoch_*.h5")),
        key=lambda x: int(re.search(r"epoch_(\d+)\.h5", x).group(1)),
    )

    # If H5 checkpoints exist, return the latest one
    if h5_files:
        latest_file = h5_files[-1]
        latest_epoch = int(re.search(r"epoch_(\d+)\.h5", latest_file).group(1))
        return latest_file, latest_epoch

    # Fallback to TensorFlow checkpoints if no H5 found
    latest_tf_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_tf_checkpoint:
        epoch_match = re.search(r"ckpt-(\d+)", latest_tf_checkpoint)
        epoch = int(epoch_match.group(1)) if epoch_match else 0
        return latest_tf_checkpoint, epoch

    # No checkpoints found
    return None, 0


def train_two_phases(model, train_dataset, val_dataset, epochs=10, checkpoint_dir="models"):
    """Train the model on the segmentation task with two phases."""

    # Phase 1: Train with frozen encoder for 1 epoch
    print("\nPhase 1: Training with frozen encoder...")
    # Freeze only the encoder layers (those named "block_*")
    for layer in model.layers:
        if "block_" in layer.name:
            layer.trainable = False
        else:
            layer.trainable = True

    # Recompile the model to ensure optimizer state matches trainability
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[iou_metric],
    )

    # Create a separate checkpoint directory for phase 1
    phase1_checkpoint_dir = os.path.join(checkpoint_dir, "phase1")
    os.makedirs(phase1_checkpoint_dir, exist_ok=True)

    # Train in Phase 1
    model, history1 = train_model(
        model,
        train_dataset,
        val_dataset,
        epochs=1,
        initial_epoch=0,
        checkpoint_dir=phase1_checkpoint_dir,
        load_latest_checkpoint=False,  # Prevent loading a checkpoint in the first phase
    )

    # Phase 2: Train with unfrozen encoder for remaining epochs
    print("\nPhase 2: Training with unfrozen encoder...")
    # Unfreeze all layers
    for layer in model.layers:
        layer.trainable = True

    # Recompile the model again
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[iou_metric],
    )

    # Train in Phase 2
    # Use the main checkpoint directory for phase 2
    model, history2 = train_model(
        model,
        train_dataset,
        val_dataset,
        epochs=epochs,
        initial_epoch=1,
        checkpoint_dir=checkpoint_dir,
    )

    # Combine histories
    combined_history = {}
    for key in history1.history:
        combined_history[key] = history1.history[key] + history2.history[key]

    return model, combined_history


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", default="/mnt/f/ssl_images/data", type=str, help="Dataset folder path"
    )
    parser.add_argument(
        "--two_phases_train", action="store_true", help="Allow training in two phases"
    )
    parser.add_argument(
        "--pretrained_model", default=None, type=str, help="Path to pretrained model"
    )
    parser.add_argument("--single_channel", action="store_true", help="To use grayscale images")
    parser.add_argument(
        "--checkpoint_dir",
        default="segmentation_ckpt",
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
        trained_model, history = train_two_phases(
            model,
            train_dataset,
            val_dataset,
            checkpoint_dir=args.checkpoint_dir,
            epochs=10,  # Specify total epochs for two-phase training
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
    final_h5_path = os.path.join(args.checkpoint_dir, "final_segmentation_model.h5")
    trained_model.save_weights(final_h5_path)

    print(f"Final model saved as TF checkpoint to {final_tf_checkpoint_path}")
    print(f"Final model weights saved to {final_h5_path}")

"""
python scripts/03_segmentation_task_training.py --two_phases_train --single_channel --checkpoint_dir models/segmentation_checkpoints_no_weights
"""
