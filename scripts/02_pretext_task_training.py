# scripts/02_pretext_task_training.py

import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.resnet import ResNet18
from src.utils.metrics import SegmentationMetrics

def load_data(data_dir, task):
    """Load preprocessed data for the specified task."""
    task_dir = os.path.join(data_dir, task)
    files = os.listdir(task_dir)
    
    if task == 'inpainting':
        masked_files = sorted([f for f in files if f.startswith('masked')])
        original_files = sorted([f for f in files if f.startswith('original')])
        return [np.load(os.path.join(task_dir, f)) for f in masked_files], [np.load(os.path.join(task_dir, f)) for f in original_files]
    elif task == 'colorization':
        gray_files = sorted([f for f in files if f.startswith('gray')])
        color_files = sorted([f for f in files if f.startswith('color')])
        return [np.load(os.path.join(task_dir, f)) for f in gray_files], [np.load(os.path.join(task_dir, f)) for f in color_files]

def create_dataset(inputs, targets, batch_size):
    """Create a TensorFlow dataset from inputs and targets."""
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    return dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def train_pretext_task(task, data_dir, model, epochs=10, batch_size=32):
    """Train the model on the specified pretext task."""
    inputs, targets = load_data(data_dir, task)
    dataset = create_dataset(inputs, targets, batch_size)

    if task == 'inpainting':
        loss = tf.keras.losses.MeanSquaredError()
        metrics = [tf.keras.metrics.MeanSquaredError(), SegmentationMetrics()]
    elif task == 'colorization':
        loss = tf.keras.losses.MeanAbsoluteError()
        metrics = [tf.keras.metrics.MeanAbsoluteError()]

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics=metrics)
    
    history = model.fit(dataset, epochs=epochs, verbose=1)
    return history

if __name__ == "__main__":
    data_dir = os.path.join("/mnt/f/ssl_images/data", "processed", "coco")
    model = ResNet18()
    model.build_encoder((None, 224, 224, 3))
    model.build_decoder(model.encoder.output_shape)

    # Train on inpainting task
    print("Training on inpainting task...")
    inpainting_history = train_pretext_task('inpainting', data_dir, model)

    # Train on colorization task
    print("Training on colorization task...")
    colorization_history = train_pretext_task('colorization', data_dir, model)

    # Save the trained model
    model.save_weights(os.path.join("models", "pretrained_resnet18.h5"))

    print("Pretext task training completed successfully!")