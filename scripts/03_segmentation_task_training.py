import os
import sys
import tensorflow as tf
import tensorflow.keras.backend as K

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.resnet import ResNet18, load_encoder_weights
from src.libs.data_loading import create_dataset_segmentation

# TODO
# - Change loss
# - Add metrics
# - Fix path to saved models


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


def train_model(model, train_dataset, val_dataset, epochs=10, initial_epoch=0):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[iou_metric],
    )

    # Create callbacks for training
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0, patience=5, verbose=1, mode="auto"
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "models/segmentation_baseline_epoch_{epoch:02d}_loss_{val_loss:.2f}.h5",
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


if __name__ == "__main__":
    data_path = os.path.join("/mnt/f/ssl_images/data/")
    data_dir = os.path.join(data_path, "processed", "coco")
    pretrained_model = os.path.join("models", "checkpoints_resnet18_vgg", "best_model.h5")
    # pretrained_model = False
    two_phases_train = True  # set to True to freeze encoder in the first epoch

    model = ResNet18((224, 224, 1), mode="segmentation")
    print(model.summary())

    if pretrained_model:
        print("Loading model weights...")
        load_encoder_weights(model, pretrained_model)

    single_channel = model.input_shape[-1] == 1

    # Load data and create dataset
    print("Loading data and creating dataset...")
    train_dataset = create_dataset_segmentation(
        data_dir,
        split="train",
        batch_size=32,
        single_channel=single_channel,
    )
    val_dataset = create_dataset_segmentation(
        data_dir,
        split="val",
        batch_size=32,
        single_channel=single_channel,
    )

    # Train the model
    print("Training the model...")
    if two_phases_train:
        trained_model, history = train_two_phases(model, train_dataset, val_dataset)
    else:
        trained_model, history = train_model(model, train_dataset, val_dataset)

    # Save the model
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "no_weights_finetune_segmentation_resnet18.h5")
    trained_model.save_weights(save_path)
    print(f"Final model saved to {save_path}")
    print("Segmentation training completed successfully!")
