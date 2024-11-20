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
    y_true: tf.Tensor, y_pred: tf.Tensor, num_classes: int = 10, threshold: float = 0.5
):
    class_iou = []
    for class_idx in range(num_classes):
        # Generate binary vector from predictions
        y_pred = tf.where(y_pred > threshold, 1.0, 0.0)

        # Extract single class to compute IoU over
        y_true_single_class = y_true[..., class_idx]
        y_pred_single_class = y_pred[..., class_idx]

        # Compute IoU
        intersection = K.sum(y_true_single_class * y_pred_single_class)
        union = K.sum(y_true_single_class) + K.sum(y_pred_single_class) - intersection

        class_iou.append(K.switch(K.equal(union, 0.0), 1.0, intersection / union))

    return sum(class_iou) / len(class_iou)


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


def _gather_channels(x, indexes, **kwargs):
    """Slice tensor along channels axis by given indexes"""
    if tf.keras.backend.image_data_format() == 'channels_last':
        x = tf.keras.permute_dimensions(x, (3, 0, 1, 2))
        x = tf.keras.gather(x, indexes)
        x = tf.keras.permute_dimensions(x, (1, 2, 3, 0))
    else:
        x = tf.keras.permute_dimensions(x, (1, 0, 2, 3))
        x = tf.keras.gather(x, indexes)
        x = tf.keras.permute_dimensions(x, (1, 0, 2, 3))
    return x

def gather_channels(*xs, indexes=None, **kwargs):
    """Slice tensors along channels axis by given indexes"""
    if indexes is None:
        return xs
    elif isinstance(indexes, (int)):
        indexes = [indexes]
    xs = [_gather_channels(x, indexes=indexes, **kwargs) for x in xs]
    return xs


def round_if_needed(x, threshold, **kwargs):
    if threshold is not None:
        x = tf.keras.greater(x, threshold)
        x = tf.keras.cast(x, tf.keras.floatx())
    return x


def average(x, per_image=False, class_weights=None, **kwargs):
    if per_image:
        x = tf.math.reduce_mean(x, axis=0)
    if class_weights is not None:
        x = x * class_weights
    return tf.math.reduce_mean(x)


def get_reduce_axes(per_image, **kwargs):
    axes = [1, 2] if tf.keras.backend.image_data_format() == 'channels_last' else [2, 3]
    if not per_image:
        axes.insert(0, 0)
    return axes


def iou_score(gt, pr, class_weights=1., class_indexes=None, smooth=1e-5, per_image=False, threshold=None, **kwargs):

    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)
    #pr = round_if_needed(pr, threshold, **kwargs)
    class_indices = tf.argmax(pr, axis=-1)
    pr = tf.one_hot(class_indices, depth=11)
    pr = tf.cast(pr, dtype=gt.dtype)

    axes = get_reduce_axes(per_image, **kwargs)

    # score calculation
    intersection = tf.math.reduce_sum(gt * pr, axis=axes)
    union = tf.math.reduce_sum(gt + pr, axis=axes) - intersection

    score = (intersection + smooth) / (union + smooth)
    score = average(score, per_image, class_weights, **kwargs)

    return score


class IOUScore:

    def __init__(
            self,
            class_weights=None,
            class_indexes=None,
            threshold=None,
            per_image=False,
            smooth=1e-5,
            name=None,
    ):
        name = name or 'iou_score'
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.threshold = threshold
        self.per_image = per_image
        self.smooth = smooth

    def __call__(self, gt, pr):
        return iou_score(
            gt,
            pr,
            class_weights=self.class_weights,
            class_indexes=self.class_indexes,
            smooth=self.smooth,
            per_image=self.per_image,
            threshold=self.threshold
        )

class FScore:

    def __init__(
            self,
            beta=1,
            class_weights=None,
            class_indexes=None,
            threshold=None,
            per_image=False,
            smooth=1e-5,
            name=None,
    ):
        name = name or 'f{}-score'.format(beta)
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.threshold = threshold
        self.per_image = per_image
        self.smooth = smooth

    def __call__(self, gt, pr):
        return f_score(
            gt,
            pr,
            beta=self.beta,
            class_weights=self.class_weights,
            class_indexes=self.class_indexes,
            smooth=self.smooth,
            per_image=self.per_image,
            threshold=self.threshold,
        )

def f_score(gt, pr, beta=1, class_weights=1, class_indexes=None, smooth=1e-5, per_image=False, threshold=None,
            **kwargs):


    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)
    class_indices = tf.argmax(pr, axis=-1)
    pr = tf.one_hot(class_indices, depth=11)
    pr = tf.cast(pr, dtype=gt.dtype)
    axes = get_reduce_axes(per_image, **kwargs)

    # calculate score
    tp = tf.math.reduce_sum(gt * pr, axis=axes)
    fp = tf.math.reduce_sum(pr, axis=axes) - tp
    fn = tf.math.reduce_sum(gt, axis=axes) - tp

    score = ((1 + beta ** 2) * tp + smooth) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = average(score, per_image, class_weights, **kwargs)

    return score



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
    class_weights = np.array([0.5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0.5])
    #class_weights = np.array([1., 5., 15., 11., 10., 8., 10., 5., 6., 14., 0.5])
    dice_loss = sm.losses.DiceLoss(class_weights=class_weights) 
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    metrics = [IOUScore(), FScore()]

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=total_loss,
        #loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=metrics,
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
    parser.add_argument("--data_path", default="/mnt/f/ssl_images/data", type=str, help="Dataset folder path")
    parser.add_argument("--two_phases_train", action='store_true', help="Allow training in two phases")
    parser.add_argument("--pretrained_model", default=None, type=str, help="Path to pretrained model")
    parser.add_argument("--single_channel", action='store_true', help="To use grayscale images")
    parser.add_argument("--checkpoint_dir", default='segmentation_ckpt', type=str, help="Prefix to the saved model path")
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
            model, train_dataset, val_dataset, checkpoint_dir=args.checkpoint_dir
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
