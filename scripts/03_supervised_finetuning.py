# scripts/03_supervised_finetuning.py

import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import xml.etree.ElementTree as ET

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.resnet import ResNet18
from src.libs.data_processing import augment_image

def load_pascal_voc_data(data_dir):
    """Load preprocessed Pascal VOC data."""
    images = []
    labels = []
    class_names = set()

    image_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'labels')

    for file in tqdm(os.listdir(image_dir), desc="Loading Pascal VOC data"):
        if file.endswith('.npy'):
            # Load image
            img = np.load(os.path.join(image_dir, file))
            images.append(img)

            # Load and parse XML annotation
            xml_file = os.path.join(label_dir, f"{os.path.splitext(file)[0]}.xml")
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get all object names in the image
            objects = [obj.find('name').text for obj in root.findall('.//object')]
            class_names.update(objects)
            labels.append(objects)

    # Create a mapping from class names to indices
    class_to_index = {name: i for i, name in enumerate(sorted(class_names))}

    # Convert labels to one-hot encoding
    one_hot_labels = []
    for label in labels:
        one_hot = np.zeros(len(class_to_index))
        for obj in label:
            one_hot[class_to_index[obj]] = 1
        one_hot_labels.append(one_hot)

    return np.array(images), np.array(one_hot_labels), class_to_index

def create_dataset(images, labels, batch_size=32):
    """Create a TensorFlow dataset for training."""
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=len(images))
    dataset = dataset.map(lambda x, y: (augment_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def fine_tune_model(model, dataset, num_classes, epochs=20):
    """Fine-tune the model on the Pascal VOC dataset."""
    fine_tuned_model = model.fine_tune(num_classes)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

    fine_tuned_model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    fine_tuned_model.fit(dataset, epochs=epochs)

    return fine_tuned_model

if __name__ == "__main__":
    data_dir = os.path.join('data', 'processed', 'pascal_voc')
    pretext_model_dir = os.path.join('models', 'pretext')
    output_dir = os.path.join('models', 'finetuned')
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    images, labels, class_to_index = load_pascal_voc_data(data_dir)
    dataset = create_dataset(images, labels)

    # Load pre-trained model (you can choose either inpainting or colorization)
    model = ResNet18()
    model.build_encoder((None, 224, 224, 3))
    model.build_decoder(model.encoder.output_shape)
    model.load_weights(os.path.join(pretext_model_dir, 'inpainting_model_weights.h5'))

    # Fine-tune the model
    fine_tuned_model = fine_tune_model(model, dataset, num_classes=len(class_to_index))

    # Save the fine-tuned model
    fine_tuned_model.save_weights(os.path.join(output_dir, 'finetuned_model_weights.h5'))

    # Save the class_to_index mapping
    np.save(os.path.join(output_dir, 'class_to_index.npy'), class_to_index)

    print("Supervised fine-tuning completed successfully!")