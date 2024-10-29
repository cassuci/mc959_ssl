import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
import sys
import glob
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.resnet import ResNet18, ResNet50


def get_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint file and extract its epoch number."""
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    # Look for epoch checkpoints
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.h5"))
    
    if not checkpoint_files:
        return None, 0
        
    # Extract epoch numbers and find the latest one
    epoch_numbers = []
    for f in checkpoint_files:
        match = re.search(r'model_epoch_(\d+)\.h5', f)
        if match:
            epoch_numbers.append((int(match.group(1)), f))
    
    if not epoch_numbers:
        return None, 0
        
    # Get the latest epoch checkpoint
    latest_epoch, latest_file = max(epoch_numbers, key=lambda x: x[0])
    
    return latest_file, latest_epoch


def get_file_paths(data_dir):
    """Get file paths for colorization task and split into train/val sets."""
    files = os.listdir(data_dir)

    gray_files = sorted([os.path.join(data_dir, f) for f in files if f.startswith("gray")])
    color_files = sorted([os.path.join(data_dir, f) for f in files if f.startswith("color")])

    # Verify matching pairs
    assert len(gray_files) == len(color_files), "Mismatched number of gray and color images"

    # Calculate split indices (80% train, 20% validation)
    total_samples = len(gray_files)
    train_size = int(0.8 * total_samples)

    # Shuffle the files with a fixed seed for reproducibility
    indices = np.random.RandomState(42).permutation(total_samples)
    gray_files = np.array(gray_files)[indices]
    color_files = np.array(color_files)[indices]

    # Split the files
    train_gray = gray_files[:train_size].tolist()
    train_color = color_files[:train_size].tolist()
    val_gray = gray_files[train_size:].tolist()
    val_color = color_files[train_size:].tolist()

    print(f"Training samples: {len(train_gray)}")
    print(f"Validation samples: {len(val_gray)}")

    return (train_gray, train_color), (val_gray, val_color)


def load_image(file_path):
    """Loads and preprocesses an image for colorization."""
    # Convert the EagerTensor back to a numpy string
    file_path = file_path.numpy().decode("utf-8")

    try:
        # Load the .npy file content
        img = np.load(file_path)

        # Preprocess the image
        img = tf.cast(img, tf.float32) / 255.0

        if "gray" in file_path:
            # Ensure grayscale input is 1 channel
            if len(img.shape) == 3 and img.shape[-1] == 3:
                img = tf.image.rgb_to_grayscale(img)

        return img
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None


def create_dataset(gray_files, color_files, batch_size, shuffle=True):
    """Creates a TensorFlow dataset for colorization."""
    # Create datasets from file paths
    gray_dataset = tf.data.Dataset.from_tensor_slices(gray_files)
    color_dataset = tf.data.Dataset.from_tensor_slices(color_files)

    # Zip the datasets together
    dataset = tf.data.Dataset.zip((gray_dataset, color_dataset))

    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    # Function to load both images
    def load_image_pair(gray_path, color_path):
        gray_img = tf.py_function(load_image, [gray_path], tf.float32)
        color_img = tf.py_function(load_image, [color_path], tf.float32)

        # Set shapes explicitly
        gray_img.set_shape([224, 224, 1])
        color_img.set_shape([224, 224, 3])

        return gray_img, color_img

    # Map the loading function and filter out None values
    dataset = dataset.map(load_image_pair, num_parallel_calls=tf.data.AUTOTUNE)

    # Important: Repeat the dataset for multiple epochs
    dataset = dataset.repeat()

    # Batch and prefetch
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


class TrainingProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback to track and save training progress."""

    def __init__(self, checkpoint_dir="checkpoints", save_freq=5):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.save_freq = save_freq
        self.best_val_loss = float("inf")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Load existing history if it exists
        history_path = os.path.join(checkpoint_dir, "training_history.npy")
        if os.path.exists(history_path):
            try:
                self.history = np.load(history_path, allow_pickle=True).item()
            except:
                self.history = self._initialize_history()
        else:
            self.history = self._initialize_history()

        # Load best validation loss if it exists
        best_loss_path = os.path.join(checkpoint_dir, "best_val_loss.npy")
        if os.path.exists(best_loss_path):
            self.best_val_loss = float(np.load(best_loss_path))

    def _initialize_history(self):
        return {
            "loss": [],
            "val_loss": [],
            "mean_absolute_error": [],
            "val_mean_absolute_error": [],
        }

    def on_epoch_end(self, epoch, logs=None):
        # Update history
        for metric in self.history.keys():
            if metric in logs:
                self.history[metric].append(logs[metric])

        # Save periodic checkpoint
        if (epoch + 1) % self.save_freq == 0:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch + 1:03d}.h5")
            self.model.save_weights(checkpoint_path)
            print(f"\nSaved periodic checkpoint for epoch {epoch + 1}")

        # Save best model
        if logs.get("val_loss", float("inf")) < self.best_val_loss:
            self.best_val_loss = logs["val_loss"]
            best_model_path = os.path.join(self.checkpoint_dir, "best_model.h5")
            self.model.save_weights(best_model_path)
            # Save best validation loss
            np.save(os.path.join(self.checkpoint_dir, "best_val_loss.npy"), self.best_val_loss)
            print(f"\nNew best model saved with validation loss: {self.best_val_loss:.6f}")

        # Save training history
        history_path = os.path.join(self.checkpoint_dir, "training_history.npy")
        np.save(history_path, self.history)


class VGGPerceptualLoss(tf.keras.Model):
    def __init__(self, resize_inputs=True):
        super().__init__()
        # Load VGG19 pretrained on ImageNet
        vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
        vgg.trainable = False

        # We'll use these activation layers for perceptual loss
        output_layers = ["block1_conv2", "block2_conv2", "block3_conv4", "block4_conv4"]

        outputs = [vgg.get_layer(name).output for name in output_layers]
        self.model = tf.keras.Model([vgg.input], outputs)
        self.resize_inputs = resize_inputs

    def call(self, inputs):
        # Preprocessing for VGG
        x = tf.keras.applications.vgg19.preprocess_input(inputs * 255.0)
        return self.model(x)


class ColorizationLoss(tf.keras.losses.Loss):
    def __init__(self, vgg_weight=1.0, l1_weight=1.0):
        super().__init__()
        self.vgg_model = VGGPerceptualLoss()
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.vgg_weight = vgg_weight
        self.l1_weight = l1_weight

    def call(self, y_true, y_pred):
        # L1 loss
        l1_loss = self.mae(y_true, y_pred)

        # VGG perceptual loss
        vgg_true = self.vgg_model(y_true)
        vgg_pred = self.vgg_model(y_pred)

        perceptual_loss = 0
        for pt, pp in zip(vgg_true, vgg_pred):
            perceptual_loss += tf.reduce_mean(tf.abs(pt - pp))

        # Combine losses
        total_loss = (self.l1_weight * l1_loss) + (self.vgg_weight * perceptual_loss)
        return total_loss


def train_colorization(data_dir, model, epochs=100, batch_size=16, checkpoint_dir=None):
    """Main training function for colorization with validation and perceptual loss."""
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join("models", "checkpoints")
    
    # Find latest checkpoint and epoch
    latest_checkpoint, initial_epoch = get_latest_checkpoint(checkpoint_dir)
    
    if latest_checkpoint:
        print(f"Found checkpoint at epoch {initial_epoch}, resuming training...")
        model.load_weights(latest_checkpoint)
    else:
        print("No checkpoint found, starting training from scratch...")
        initial_epoch = 0

    # Get train and validation file paths
    (train_gray, train_color), (val_gray, val_color) = get_file_paths(data_dir)

    # Create training and validation datasets
    train_dataset = create_dataset(train_gray, train_color, batch_size, shuffle=True)
    val_dataset = create_dataset(val_gray, val_color, batch_size, shuffle=False)

    # Calculate steps per epoch
    train_steps = len(train_gray) // batch_size
    val_steps = len(val_gray) // batch_size

    # Print training configuration
    print("\nTraining Configuration:")
    print(f"Total training samples: {len(train_gray)}")
    print(f"Total validation samples: {len(val_gray)}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {train_steps}")
    print(f"Validation steps: {val_steps}")
    print(f"Starting from epoch: {initial_epoch + 1}")
    print("=" * 50, "\n")

    # Initialize loss function with weights
    loss_fn = ColorizationLoss(vgg_weight=0.1, l1_weight=100.0)

    # Custom metric for monitoring L1 loss only
    def l1_metric(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))

    # Compile model with custom loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
        loss=loss_fn, 
        metrics=[l1_metric]
    )

    # Add callbacks
    callbacks = [
        TrainingProgressCallback(checkpoint_dir=checkpoint_dir, save_freq=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
    ]

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )

    return history


if __name__ == "__main__":
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Set up paths
    data_dir = os.path.join("/mnt/f/ssl_images/data", "processed", "coco", "colorization")
    checkpoint_dir = os.path.join("models", "checkpoints")

    # Initialize model
    model = ResNet50((224, 224, 1))

    print("Training colorization model...")
    history = train_colorization(data_dir, model, epochs=100, batch_size=24, checkpoint_dir=checkpoint_dir)

    # Save the final model
    save_path = os.path.join("models", "colorization_model_resnet50.h5")
    model.save_weights(save_path)
    print(f"Final model saved to {save_path}")
    print("Colorization training completed successfully!")