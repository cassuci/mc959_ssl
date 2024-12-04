# scripts/03_baseline.py
import os
import sys
import tensorflow as tf
import argparse
import numpy as np

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.resnet import ResNet18, ResNet50, load_encoder_weights
from src.libs.data_loading import create_dataset_classification


def weighted_binary_cross_entropy(weights: dict, from_logits: bool = False):
    """
    Creates a weighted binary cross-entropy loss function.
    
    Args:
        weights (dict): Dictionary containing weights for class 0 and class 1.
        from_logits (bool): Whether the input is logits or probabilities.
    
    Returns:
        function: A loss function that calculates weighted binary cross-entropy.
    """
    assert 0 in weights
    assert 1 in weights

    def weighted_cross_entropy_fn(y_true, y_pred):
        # Compute the weighted binary cross-entropy
        tf_y_true = tf.cast(y_true, dtype=y_pred.dtype)
        tf_y_pred = tf.cast(y_pred, dtype=y_pred.dtype)
        weights_v = tf.where(tf.equal(tf_y_true, 1), weights[1], weights[0])
        weights_v = tf.cast(weights_v, dtype=y_pred.dtype)
        ce = tf.keras.backend.binary_crossentropy(tf_y_true, tf_y_pred, from_logits=from_logits)
        loss = tf.keras.backend.mean(tf.multiply(ce, weights_v))
        return loss

    return weighted_cross_entropy_fn


class TrainingProgressCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to monitor and save training progress.
    
    Args:
        checkpoint_dir (str): Directory to save checkpoints and training history.
        save_freq (int): Frequency (in epochs) to save periodic checkpoints.
    
    Attributes:
        checkpoint_dir (str): Directory to save model checkpoints.
        save_freq (int): Frequency for checkpoint saving.
        best_val_loss (float): Best validation loss encountered during training.
        history (dict): Dictionary to store training metrics history.
    """
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
        """
        Initializes an empty history dictionary.
        
        Returns:
            dict: Dictionary with keys for loss, metrics, and learning rate.
        """
        return {"loss": [], "val_loss": [], "iou_metric": [], "val_iou_metric": [], "lr": []}

    def on_epoch_end(self, epoch, logs=None):
        """
        Save loss, metrics, learning rate and model checkpoint
        
        Args:
            epoch (int): Current epoch number.
            logs (dict): Dictionary containing training metrics.
        """
        logs = logs or {}
        # Log metrics 
        for metric in self.history.keys():
            if metric in logs:
                self.history[metric].append(logs[metric])
            elif metric == "lr":
                # Manually get current learning rate
                lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
                self.history["lr"].append(lr)

        # Save periodic checkpoint
        if (epoch + 1) % self.save_freq == 0:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"classification_model_epoch_{epoch + 1:03d}.weights.h5"
            )
            self.model.save_weights(checkpoint_path)
            print(f"\nSaved periodic checkpoint for epoch {epoch + 1}")

        # Save best model if val loss improved
        if logs.get("val_loss", float("inf")) < self.best_val_loss:
            self.best_val_loss = logs["val_loss"]
            best_model_path = os.path.join(
                self.checkpoint_dir, "best_classification_model.weights.h5"
            )
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
    checkpoint_dir="segmentation_ckpt",
):
    """
    Compiles and trains the model using the specified datasets and settings.
    
    Args:
        model (tf.keras.Model): Model to be trained.
        train_dataset (tf.data.Dataset): Dataset for training.
        val_dataset (tf.data.Dataset): Dataset for validation.
        epochs (int): Total number of epochs to train.
        initial_epoch (int): Starting epoch for training.
        checkpoint_dir (str): Directory for saving checkpoints.
    
    Returns:
        tuple: Trained model and training history.
    """
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        # loss=weighted_binary_cross_entropy(weights={0: 0.1, 1: 1.}, from_logits=False),
        # In *my* experiments for the baseline training, binary crossentroy *without* weights is the best loss
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.1),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(multi_label=True, name="auc_roc"),
            tf.keras.metrics.AUC(curve="PR", name="average_precision"),
        ],
    )

    # Create callbacks for training
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0, patience=5, verbose=1, mode="auto"
    )
    checkpoint = TrainingProgressCallback(checkpoint_dir=checkpoint_dir)
    reduceLRcallback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, verbose=1, mode="auto"
    )

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        initial_epoch=initial_epoch,
        epochs=epochs,
        verbose=1,
        callbacks=[early_stopping, checkpoint, reduceLRcallback],
    )

    return model, history


def train_two_phases(model, train_dataset, val_dataset, epochs=10):
    """
    Trains the model in two phases: with the encoder frozen, then unfrozen.
    
    Args:
        model (tf.keras.Model): Model to be trained.
        train_dataset (tf.data.Dataset): Dataset for training.
        val_dataset (tf.data.Dataset): Dataset for validation.
        epochs (int): Total number of epochs for training.

    Returns:
        tuple: Trained model and combined training history from both phases.
    """

    # Phase 1: Train with frozen encoder for 1 epoch
    print("\nPhase 1: Training with frozen encoder...")
    # Freeze encoder layers
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.Dense):
            layer.trainable = False

    model, history1 = train_model(model, train_dataset, val_dataset, epochs=1, initial_epoch=0)

    # Phase 2: Train with unfrozen encoder for remaining epochs
    print("\nPhase 2: Training with unfrozen encoder...")
    # Unfreeze encoder layers
    for layer in model.layers:
        layer.trainable = True

    model, history2 = train_model(
        model, train_dataset, val_dataset, epochs=epochs, initial_epoch=1
    )

    # Combine histories
    combined_history = {}
    for key in history1.history:
        try:
            combined_history[key] = history1.history[key] + history2.history[key]
        except:
            combined_history[key] = history1.history[key] + history2.history[key + "_1"]

    return model, combined_history


def get_args():
    """
    Parses and returns command-line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", default="data", type=str, help="Dataset folder path"
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
        default="models/classification_ckpt",
        type=str,
        help="Prefix to the saved model path",
    )
    parser.add_argument("--classification_classes", nargs="+", default=["person"])
    return parser.parse_args()


if __name__ == "__main__":
    """
    Main function for training a classification model.
    
    Steps:
        1. Parse command-line arguments.
        2. Initialize the model.
        3. Load datasets for training and validation.
        4. Train the model with specified settings.
        5. Save the trained model.
    """
    args = get_args()
    data_dir = os.path.join(args.data_path, "processed", "pascal_voc")
    metadata_dir = os.path.join(args.data_path, "pascal_voc", "ImageSets", "Main")
    num_classes = len(args.classification_classes) if args.classification_classes else 20

    # Initialize the model
    if args.single_channel:
        model = ResNet18((224, 224, 1), mode="classification", num_classes=num_classes)
    else:
        model = ResNet18((224, 224, 3), mode="classification", num_classes=num_classes)
    print(model.summary())

    # Load pretrained weights if specified
    if args.pretrained_model:
        print("Loading model weights...")
        load_encoder_weights(model, args.pretrained_model)

    # Create train and val datasets
    print("Loading data and creating dataset...")
    train_dataset = create_dataset_classification(
        data_dir,
        split_name='train',
        batch_size=32,
        single_channel=args.single_channel,
        classes=args.classification_classes,
    )
    val_dataset = create_dataset_classification(
        data_dir,
        split_name='val',
        batch_size=32,
        single_channel=args.single_channel,
        classes=args.classification_classes,
    )

    # Train the model
    print("Training the model...")
    if args.two_phases_train:
        trained_model, history = train_two_phases(model, train_dataset, val_dataset)
    else:
        trained_model, history = train_model(
            model, train_dataset, val_dataset, epochs=20, checkpoint_dir=args.checkpoint_dir
        )

    # Save the model
    final_h5_path = os.path.join(args.checkpoint_dir, "final_classification_model.h5")
    trained_model.save_weights(final_h5_path)
    print(f"Final model weights saved to {final_h5_path}")
    print("Supervised training completed successfully!")

"""
For person classification only:
 python scripts/03_classification_task_training.py --classification_classes person 
To classify all 20 classes:
 python scripts/03_classification_task_training.py
"""
