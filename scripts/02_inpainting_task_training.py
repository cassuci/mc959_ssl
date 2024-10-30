import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
import sys
import glob
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.resnet import ResNet18


def get_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint file and extract its epoch number."""
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.h5"))
    
    if not checkpoint_files:
        return None, 0
        
    epoch_numbers = []
    for f in checkpoint_files:
        match = re.search(r'model_epoch_(\d+)\.h5', f)
        if match:
            epoch_numbers.append((int(match.group(1)), f))
    
    if not epoch_numbers:
        return None, 0
        
    latest_epoch, latest_file = max(epoch_numbers, key=lambda x: x[0])
    return latest_file, latest_epoch


def get_file_paths(data_dir):
    """Get file paths for inpainting task and split into train/val sets."""
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")

    files = os.listdir(data_dir)

    masked_files = sorted([os.path.join(data_dir, f) for f in files if f.startswith("masked")])
    original_files = sorted([os.path.join(data_dir, f) for f in files if f.startswith("original")])
    mask_files = sorted([os.path.join(data_dir, f) for f in files if f.startswith("mask")])

    if len(masked_files) != len(original_files) or len(masked_files) != len(mask_files):
        raise ValueError("Number of masked, original, and mask files don't match")

    # Calculate split indices (80% train, 20% validation)
    total_samples = len(masked_files)
    train_size = int(0.8 * total_samples)

    # Shuffle the files with fixed seed for reproducibility
    indices = np.random.RandomState(42).permutation(total_samples)
    masked_files = np.array(masked_files)[indices]
    original_files = np.array(original_files)[indices]
    mask_files = np.array(mask_files)[indices]

    # Split the files
    train_masked = masked_files[:train_size].tolist()
    train_original = original_files[:train_size].tolist()
    train_mask = mask_files[:train_size].tolist()
    val_masked = masked_files[train_size:].tolist()
    val_original = original_files[train_size:].tolist()
    val_mask = mask_files[train_size:].tolist()

    print(f"Training samples: {len(train_masked)}")
    print(f"Validation samples: {len(val_masked)}")

    return (train_masked, train_original, train_mask), (val_masked, val_original, val_mask)


def load_image(file_path):
    """Loads and preprocesses an image for inpainting."""
    file_path = file_path.numpy().decode("utf-8")

    try:
        img = np.load(file_path)
        img = tf.cast(img, tf.float32) / 255.0
        return img
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None


def create_dataset(masked_files, original_files, mask_files, batch_size, shuffle=True):
    """Creates a TensorFlow dataset for inpainting."""
    # Create datasets from file paths
    masked_dataset = tf.data.Dataset.from_tensor_slices(masked_files)
    original_dataset = tf.data.Dataset.from_tensor_slices(original_files)
    mask_dataset = tf.data.Dataset.from_tensor_slices(mask_files)

    # Zip the datasets together
    dataset = tf.data.Dataset.zip((masked_dataset, original_dataset, mask_dataset))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    def load_image_triplet(masked_path, original_path, mask_path):
        masked_img = tf.py_function(load_image, [masked_path], tf.float32)
        original_img = tf.py_function(load_image, [original_path], tf.float32)
        mask = tf.py_function(load_image, [mask_path], tf.float32)

        # Set shapes explicitly
        masked_img.set_shape([224, 224, 3])
        original_img.set_shape([224, 224, 3])
        mask.set_shape([224, 224, 1])

        return masked_img, original_img, mask

    dataset = dataset.map(load_image_triplet, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


class VGGPerceptualLoss(tf.keras.Model):
    def __init__(self, resize_inputs=True):
        super().__init__()
        vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
        vgg.trainable = False

        output_layers = ["block1_conv2", "block2_conv2", "block3_conv4", "block4_conv4"]
        outputs = [vgg.get_layer(name).output for name in output_layers]
        self.model = tf.keras.Model([vgg.input], outputs)
        self.resize_inputs = resize_inputs

    def call(self, inputs):
        x = tf.keras.applications.vgg19.preprocess_input(inputs * 255.0)
        return self.model(x)


class MaskedInpaintingLoss(tf.keras.losses.Loss):
    def __init__(self, vgg_weight=1.0, l1_weight=1.0):
        super().__init__()
        self.vgg_model = VGGPerceptualLoss()
        self.mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
        self.vgg_weight = vgg_weight
        self.l1_weight = l1_weight
        
    def call(self, y_true, y_pred, mask):
        # Compute L1 loss only on masked regions
        pixel_l1_loss = self.mae(y_true, y_pred)
        masked_l1_loss = tf.reduce_sum(pixel_l1_loss * mask) / (tf.reduce_sum(mask) + 1e-8)
        
        # VGG perceptual loss on masked regions
        vgg_true = self.vgg_model(y_true)
        vgg_pred = self.vgg_model(y_pred)
        
        perceptual_loss = 0
        for pt, pp in zip(vgg_true, vgg_pred):
            # Resize mask to match feature map size
            curr_mask = tf.image.resize(mask, pt.shape[1:3], method='nearest')
            curr_mask = tf.repeat(curr_mask, pt.shape[-1], axis=-1)
            
            # Compute masked feature difference
            feat_diff = tf.square(pt - pp) * curr_mask
            perceptual_loss += tf.reduce_sum(feat_diff) / (tf.reduce_sum(curr_mask) + 1e-8)
            
        # Combine losses
        total_loss = (self.l1_weight * masked_l1_loss) + (self.vgg_weight * perceptual_loss)
        return total_loss


class TrainingProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback to track and save training progress."""

    def __init__(self, checkpoint_dir="checkpoints", save_freq=5):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.save_freq = save_freq
        self.best_val_loss = float("inf")
        os.makedirs(checkpoint_dir, exist_ok=True)

        history_path = os.path.join(checkpoint_dir, "training_history.npy")
        if os.path.exists(history_path):
            try:
                self.history = np.load(history_path, allow_pickle=True).item()
            except:
                self.history = self._initialize_history()
        else:
            self.history = self._initialize_history()

        best_loss_path = os.path.join(checkpoint_dir, "best_val_loss.npy")
        if os.path.exists(best_loss_path):
            self.best_val_loss = float(np.load(best_loss_path))

    def _initialize_history(self):
        return {
            "loss": [],
            "val_loss": [],
            "masked_mae": [],
            "val_masked_mae": [],
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
            np.save(os.path.join(self.checkpoint_dir, "best_val_loss.npy"), self.best_val_loss)
            print(f"\nNew best model saved with validation loss: {self.best_val_loss:.6f}")

        # Save training history
        history_path = os.path.join(self.checkpoint_dir, "training_history.npy")
        np.save(history_path, self.history)


def masked_mae_metric(y_true, y_pred, mask):
    """Compute MAE only on masked regions."""
    error = tf.abs(y_true - y_pred)
    masked_error = error * mask
    return tf.reduce_sum(masked_error) / (tf.reduce_sum(mask) + 1e-8)


def train_inpainting(data_dir, model, epochs=100, batch_size=16, checkpoint_dir=None):
    """Main training function for inpainting with validation and perceptual loss."""
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
    (train_masked, train_original, train_mask), (val_masked, val_original, val_mask) = get_file_paths(data_dir)

    # Create datasets
    train_dataset = create_dataset(train_masked, train_original, train_mask, batch_size, shuffle=True)
    val_dataset = create_dataset(val_masked, val_original, val_mask, batch_size, shuffle=False)

    # Calculate steps per epoch
    train_steps = len(train_masked) // batch_size
    val_steps = len(val_masked) // batch_size

    # Print training configuration
    print("\nTraining Configuration:")
    print(f"Total training samples: {len(train_masked)}")
    print(f"Total validation samples: {len(val_masked)}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {train_steps}")
    print(f"Validation steps: {val_steps}")
    print(f"Starting from epoch: {initial_epoch + 1}")
    print("=" * 50, "\n")

    # Initialize loss function
    loss_fn = MaskedInpaintingLoss(vgg_weight=0.5, l1_weight=1.0)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=loss_fn,
        metrics=[masked_mae_metric]
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
    data_dir = os.path.join("/mnt/f/ssl_images/data", "processed", "coco", "inpainting")
    checkpoint_dir = os.path.join("models", "checkpoints")

    # Initialize model
    model = ResNet18((224, 224, 3))

    print("Training inpainting model...")
    history = train_inpainting(data_dir, model, epochs=100, batch_size=24, checkpoint_dir=checkpoint_dir)

    # Save the final model
    save_path = os.path.join("models", "inpainting_model_final.h5")
    model.save_weights(save_path)
    print(f"Final model saved to {save_path}")
    print("Inpainting training completed successfully!")