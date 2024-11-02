# scripts/03_supervised_finetuning.py

import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.resnet import ResNet18
from src.utils.metrics import classification_metrics
from src.libs.data_loading import load_classification_data, create_dataset


def fine_tune_model(model, dataset, num_classes, epochs=10):
    """Fine-tune the model on the classification task."""
    fine_tuned_model = model.fine_tune(num_classes)

    fine_tuned_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    history = fine_tuned_model.fit(dataset, epochs=epochs, verbose=1)
    return fine_tuned_model, history


if __name__ == "__main__":
    data_dir = os.path.join("data", "processed")
    model = ResNet18()
    model.build_encoder((None, 224, 224, 3))
    model.build_decoder(model.encoder.output_shape)

    # Load pre-trained weights
    model.load_weights(os.path.join("models", "pretrained_resnet18.h5"))

    # Load classification data
    images, labels = load_classification_data(data_dir)
    num_classes = len(np.unique(labels))

    # Create dataset
    dataset = create_dataset(images, labels, batch_size=32)

    # Fine-tune the model
    print("Fine-tuning the model...")
    fine_tuned_model, history = fine_tune_model(model, dataset, num_classes)

    # Save the fine-tuned model
    fine_tuned_model.save_weights(os.path.join("models", "fine_tuned_resnet18.h5"))

    print("Supervised fine-tuning completed successfully!")
