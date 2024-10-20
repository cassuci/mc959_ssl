import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.resnet import ResNet18
from src.utils.metrics import SegmentationMetrics

def load_file_paths(data_dir, task):
    """Load file paths for the specified task."""
    task_dir = os.path.join(data_dir, task)
    files = os.listdir(task_dir)
   
    if task == 'inpainting':
        masked_files = sorted([os.path.join(task_dir, f) for f in files if f.startswith('masked')])
        original_files = sorted([os.path.join(task_dir, f) for f in files if f.startswith('original')])
        return masked_files, original_files
    elif task == 'colorization':
        gray_files = sorted([os.path.join(task_dir, f) for f in files if f.startswith('gray')])
        color_files = sorted([os.path.join(task_dir, f) for f in files if f.startswith('color')])
        return gray_files, color_files

def load_batch(file_paths, batch_size, start_idx):
    """Load a batch of images from file paths."""
    batch = []
    for i in range(start_idx, min(start_idx + batch_size, len(file_paths))):
        batch.append(np.load(file_paths[i]))
    return np.array(batch)

def create_dataset(input_paths, target_paths, batch_size):
    """Create a TensorFlow dataset from file paths."""
    total_samples = len(input_paths)
    
    def generator():
        for start_idx in range(0, total_samples, batch_size):
            input_batch = load_batch(input_paths, batch_size, start_idx)
            target_batch = load_batch(target_paths, batch_size, start_idx)
            yield input_batch, target_batch
    
    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

def train_pretext_task(task, data_dir, model, epochs=10, batch_size=32):
    """Train the model on the specified pretext task."""
    input_paths, target_paths = load_file_paths(data_dir, task)
    dataset = create_dataset(input_paths, target_paths, batch_size)
    
    if task == 'inpainting':
        loss = tf.keras.losses.MeanSquaredError()
        metrics = [tf.keras.metrics.MeanSquaredError(), SegmentationMetrics()]
    elif task == 'colorization':
        loss = tf.keras.losses.MeanAbsoluteError()
        metrics = [tf.keras.metrics.MeanAbsoluteError()]
    
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics=metrics)
   
    steps_per_epoch = len(input_paths) // batch_size
    history = model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1)
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