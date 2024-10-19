# scripts/02_pretext_task_training.py

import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.resnet import ResNet18
from src.libs.data_processing import augment_image

def load_data(data_dir, task):
    """Load preprocessed data for the specified task."""
    original_images = []
    task_images = []

    task_dir = os.path.join(data_dir, task)
    for file in tqdm(os.listdir(task_dir), desc=f"Loading {task} data"):
        if file.startswith('original_'):
            original_images.append(np.load(os.path.join(task_dir, file)))
        elif file.startswith(f"{task.split('_')[0]}_"):
            task_images.append(np.load(os.path.join(task_dir, file)))

    return np.array(original_images), np.array(task_images)

def create_dataset(original_images, task_images, batch_size=32):
    """Create a TensorFlow dataset for training."""
    dataset = tf.data.Dataset.from_tensor_slices((task_images, original_images))
    dataset = dataset.shuffle(buffer_size=len(original_images))
    dataset = dataset.map(lambda x, y: (augment_image(x), augment_image(y)), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def train_pretext_task(task, data_dir, output_dir, epochs=50, batch_size=32):
    """Train the model on the specified pretext task."""
    original_images, task_images = load_data(data_dir, task)
    dataset = create_dataset(original_images, task_images, batch_size)

    model = ResNet18()
    model.build_encoder((None, 224, 224, 3))
    model.build_decoder(model.encoder.output_shape)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss_fn = tf.keras.losses.MeanSquaredError()

    @tf.function
    def train_step(task_image, original_image):
        with tf.GradientTape() as tape:
            reconstructed_image = model(task_image)
            loss = loss_fn(original_image, reconstructed_image)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    for epoch in range(epochs):
        epoch_loss = tf.keras.metrics.Mean()
        for task_image_batch, original_image_batch in dataset:
            loss = train_step(task_image_batch, original_image_batch)
            epoch_loss.update_state(loss)
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss.result():.4f}")

    # Save the trained model
    model.save_weights(os.path.join(output_dir, f'{task}_model_weights.h5'))

if __name__ == "__main__":
    data_dir = os.path.join('data', 'processed', 'coco')
    output_dir = os.path.join('models', 'pretext')
    os.makedirs(output_dir, exist_ok=True)

    train_pretext_task('inpainting', data_dir, output_dir)
    train_pretext_task('colorization', data_dir, output_dir)

    print("Pretext task training completed successfully!")