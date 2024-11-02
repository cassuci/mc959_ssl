# scripts/05_baseline.py

import os
import sys
import tensorflow as tf

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.resnet import ResNet18
from src.libs.data_loading import load_classification_data, create_dataset


def fine_tune_model(model, train_dataset, val_dataset, epochs=10):
    """Fine-tune the model on the classification task."""

    # TODO Fix metrics and loss for multilabel classification
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # TODO Add callbacks

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, verbose=1)
    return model, history


if __name__ == "__main__":
    # TODO Customize paths
    data_dir = os.path.join("ssl_images/data", "processed", "pascal_voc")
    metadata_dir = os.path.join("ssl_images/data", "pascal_voc", "ImageSets", "Main")

    model = ResNet18((224, 224, 3), mode='classification')
    print(model.summary())

    # Load classification data
    print("Loading data and creating dataset...")
    train_images, train_labels = load_classification_data(data_dir, split_list_file=os.path.join(metadata_dir, 'train.txt'))
    val_images, val_labels = load_classification_data(data_dir, split_list_file=os.path.join(metadata_dir, 'val.txt'))

    # Create dataset
    train_dataset = create_dataset(train_images, train_labels, batch_size=32)
    val_dataset = create_dataset(val_images, val_labels, batch_size=32)

    # Fine-tune the model
    print("Fine-tuning the model...")
    fine_tuned_model, history = fine_tune_model(model, train_dataset, val_dataset)

    # Save the fine-tuned model
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "baseline_resnet18.h5")
    fine_tuned_model.save_weights(save_path)
    print(f"Final model saved to {save_path}")

    print("Supervised fine-tuning completed successfully!")