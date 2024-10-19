# scripts/04_evaluation.py

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.resnet import ResNet18
from src.utils.metrics import classification_metrics
from src.libs.visualization import plot_training_history, visualize_latent_space

def load_and_split_data(data_dir):
    """Load preprocessed data and split into train and test sets."""
    task_dir = os.path.join(data_dir, 'classification')
    files = os.listdir(task_dir)
    images = [np.load(os.path.join(task_dir, f)) for f in files if f.endswith('.npy')]
    
    # TODO: Load and process actual labels
    labels = np.random.randint(0, 20, len(images))  # Placeholder, replace with actual labels
    
    return train_test_split(np.array(images), labels, test_size=0.2, random_state=42)

def evaluate_model(model, x_test, y_test):
    """Evaluate the model and return predictions and metrics."""
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    metrics = classification_metrics(y_test, y_pred_classes, y_pred)
    return y_pred, metrics

if __name__ == "__main__":
    data_dir = os.path.join("data", "processed")
    
    # Load and split data
    x_train, x_test, y_train, y_test = load_and_split_data(data_dir)
    num_classes = len(np.unique(y_train))

    # Load the fine-tuned model
    model = ResNet18()
    model.build_encoder((None, 224, 224, 3))
    model.build_decoder(model.encoder.output_shape)
    fine_tuned_model = model.fine_tune(num_classes)
    fine_tuned_model.load_weights(os.path.join("models", "fine_tuned_resnet18.h5"))

    # Evaluate the model
    print("Evaluating the model...")
    y_pred, metrics = evaluate_model(fine_tuned_model, x_test, y_test)

    # Print evaluation metrics
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    # Visualize latent space
    print("Visualizing latent space...")
    visualize_latent_space(model.encoder, tf.data.Dataset.from_tensor_slices(x_test).batch(32))

    print("Evaluation completed successfully!")