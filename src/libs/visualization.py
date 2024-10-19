# src/libs/visualization.py

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE

def plot_images(images, titles=None, cols=5, figsize=(15, 15)):
    """Plot a list of images in a grid."""
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i, img in enumerate(images):
        if img.shape[-1] == 1:
            img = np.squeeze(img)
        axes[i].imshow(img)
        axes[i].axis('off')
        if titles:
            axes[i].set_title(titles[i])

    for i in range(len(images), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def plot_feature_maps(feature_maps, layer_name, num_filters=16):
    """Plot feature maps from a convolutional layer."""
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    for i, ax in enumerate(axes.flatten()):
        if i < num_filters:
            ax.imshow(feature_maps[0, :, :, i], cmap='viridis')
        ax.axis('off')
    plt.suptitle(f'Feature maps from {layer_name}')
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """Plot training and validation loss/accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def visualize_latent_space(model, dataset, num_samples=1000):
    """Visualize the latent space using t-SNE."""
    features = []
    labels = []

    for images, batch_labels in dataset.take(num_samples // 32):
        batch_features = model.predict(images)
        features.extend(batch_features)
        labels.extend(batch_labels.numpy())

    features = np.array(features)
    labels = np.array(labels)

    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title('t-SNE visualization of the latent space')
    plt.show()

def plot_segmentation_results(image, true_mask, pred_mask):
    """Plot original image, true mask, and predicted mask for segmentation tasks."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(true_mask)
    ax2.set_title('True Mask')
    ax2.axis('off')

    ax3.imshow(pred_mask)
    ax3.set_title('Predicted Mask')
    ax3.axis('off')

    plt.tight_layout()
    plt.show()