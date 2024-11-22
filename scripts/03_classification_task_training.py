# scripts/03_baseline.py
import os
import sys
import tensorflow as tf
import argparse

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.resnet import ResNet18, ResNet50, load_encoder_weights
from src.libs.data_loading import create_dataset_classification


def weighted_binary_cross_entropy(weights: dict, from_logits: bool = False):
    assert 0 in weights
    assert 1 in weights

    def weighted_cross_entropy_fn(y_true, y_pred):
        tf_y_true = tf.cast(y_true, dtype=y_pred.dtype)
        tf_y_pred = tf.cast(y_pred, dtype=y_pred.dtype)

        weights_v = tf.where(tf.equal(tf_y_true, 1), weights[1], weights[0])
        weights_v = tf.cast(weights_v, dtype=y_pred.dtype)
        ce = tf.keras.backend.binary_crossentropy(tf_y_true, tf_y_pred, from_logits=from_logits)
        loss = tf.keras.backend.mean(tf.multiply(ce, weights_v))
        return loss

    return weighted_cross_entropy_fn


def train_model(model, train_dataset, val_dataset, epochs=10, initial_epoch=0):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=weighted_binary_cross_entropy(weights={0: 1.0, 1: 9.0}, from_logits=False),
        # loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.1),
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
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "models/finetune_resnet50_new_loss/model_epoch_{epoch:02d}_loss_{val_loss:.2f}.h5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
    )
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
    """Train the model on the classification task with two phases."""

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
        default="classification_ckpt",
        type=str,
        help="Prefix to the saved model path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    data_dir = os.path.join(args.data_path, "processed", "pascal_voc")
    metadata_dir = os.path.join(args.data_path, "pascal_voc", "ImageSets", "Main")

    if args.single_channel:
        model = ResNet18((224, 224, 1), mode="classification")
    else:
        model = ResNet18((224, 224, 3), mode="classification")
    print(model.summary())

    if args.pretrained_model:
        print("Loading model weights...")
        load_encoder_weights(model, args.pretrained_model)

    # Load data and create dataset
    print("Loading data and creating dataset...")
    train_dataset = create_dataset_classification(
        data_dir,
        split_list_file=os.path.join(metadata_dir, "train.txt"),
        batch_size=32,
        single_channel=args.single_channel,
    )
    val_dataset = create_dataset_classification(
        data_dir,
        split_list_file=os.path.join(metadata_dir, "val.txt"),
        batch_size=32,
        single_channel=args.single_channel,
    )

    # Train the model
    print("Training the model...")
    if args.two_phases_train:
        fine_tuned_model, history = train_two_phases(model, train_dataset, val_dataset)
    else:
        fine_tuned_model, history = train_model(model, train_dataset, val_dataset)

    # Save the model
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", f"{args.checkpoint_dir}.h5")
    fine_tuned_model.save_weights(save_path)
    print(f"Final model saved to {save_path}")
    print("Supervised training completed successfully!")
