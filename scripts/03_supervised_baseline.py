# scripts/03_baseline.py

import os
import sys
import tensorflow as tf

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.resnet import ResNet18
from src.libs.data_loading import load_classification_data, create_dataset


def train_model(model, train_dataset, val_dataset, epochs=10):
    """Train the model on the classification task."""

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, alpha=0.6),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                      min_delta=0, 
                                                      patience=5, 
                                                      verbose=1, 
                                                      mode='auto')

    checkpoint = tf.keras.callbacks.ModelCheckpoint("models/baseline_{epoch:02d}_{val_loss:.2f}.h5", 
                                                    monitor='val_loss', 
                                                    verbose=1, 
                                                    save_best_only=True, 
                                                    save_weights_only=True,
                                                    mode='auto')

    reduceLRcallback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                            factor=0.5, 
                                                            patience=3,
                                                            verbose=1, 
                                                            mode='auto')

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, verbose=1,
                        callbacks = [early_stopping, checkpoint, reduceLRcallback])
    return model, history


if __name__ == "__main__":
    data_path = "ssl_images/data"   # "/mnt/f/ssl_images/data" if you're Gabriel 
    data_dir = os.path.join(data_path, "processed", "pascal_voc")
    metadata_dir = os.path.join(data_path, "pascal_voc", "ImageSets", "Main")

    model = ResNet18((224, 224, 3), mode='classification')
    print(model.summary())

    # Load classification data
    print("Loading data and creating dataset...")
    train_images, train_labels = load_classification_data(data_dir, split_list_file=os.path.join(metadata_dir, 'train.txt'))
    val_images, val_labels = load_classification_data(data_dir, split_list_file=os.path.join(metadata_dir, 'val.txt'))

    # Create dataset
    train_dataset = create_dataset(train_images, train_labels, batch_size=32)
    val_dataset = create_dataset(val_images, val_labels, batch_size=32)

    # Train the model
    print("Training the model...")
    fine_tuned_model, history = train_model(model, train_dataset, val_dataset)

    # Save the model
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "baseline_resnet18.h5")
    fine_tuned_model.save_weights(save_path)
    print(f"Final model saved to {save_path}")

    print("Supervised training completed successfully!")