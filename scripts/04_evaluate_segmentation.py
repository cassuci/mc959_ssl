import os
import sys
import numpy as np
import tensorflow as tf
import argparse
#import segmentation_models as sm

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.resnet import ResNet18
from src.libs.data_loading import create_dataset_segmentation

def iou_metric(
    y_true: tf.Tensor, y_pred: tf.Tensor, num_classes: int = 3, threshold: float = 0.5
) -> tf.Tensor:
    # Binarize predictions
    y_pred_bin = tf.cast(y_pred > threshold, tf.float32)

    # Initialize list to store IoU for each class
    ious = []

    for class_idx in range(num_classes):
        # Extract predictions and ground truth for the current class
        y_true_class = y_true[..., class_idx]
        y_pred_class = y_pred_bin[..., class_idx]

        # Compute intersection and union
        intersection = tf.reduce_sum(y_true_class * y_pred_class, axis=(1, 2))
        union = (
            tf.reduce_sum(y_true_class, axis=(1, 2))
            + tf.reduce_sum(y_pred_class, axis=(1, 2))
            - intersection
        )

        # Avoid division by zero
        iou = tf.where(union > 0, intersection / union, tf.ones_like(union))
        ious.append(tf.reduce_mean(iou))  # Average over batch for each class

    return ious

# def get_segmentation_loss():
#     class_weights = np.array([1, 1, 1, 0.05])
#     dice_loss = sm.losses.DiceLoss(class_weights=class_weights)
#     focal_loss = sm.losses.CategoricalFocalLoss()
#     total_loss = dice_loss + (1 * focal_loss)
#     return total_loss

def evaluate_model(model, test_dataset, num_classes=3):
    """
    Evaluate the model on test dataset and compute metrics per class
    """
    total_loss = 0
    class_ious = np.zeros(num_classes)  # Assuming 3 classes
    num_batches = 0
    
    # Compile model with same loss and metrics as training
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(), # it's wrong, but just to load the model
        metrics=[iou_metric],
    )
    
    # Evaluate over all batches
    for x_batch, y_batch in test_dataset:
        # Get predictions
        y_pred = model.predict(x_batch, verbose=0)
        
        # Compute loss
        batch_loss = model.loss(y_batch, y_pred)
        total_loss += batch_loss
        
        # Compute IoU for each class
        batch_ious = iou_metric(y_batch, y_pred, num_classes=num_classes)
        class_ious += np.array(batch_ious)
        
        num_batches += 1
    
    # Compute averages
    avg_loss = total_loss / num_batches
    avg_class_ious = class_ious / num_batches
    mean_iou = np.mean(avg_class_ious)
    
    return {
        'loss': float(avg_loss),
        'mean_iou': float(mean_iou),
        'class_ious': avg_class_ious.tolist()
    }

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", 
        default="/mnt/f/ssl_images/data", 
        type=str, 
        help="Dataset folder path"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path to the trained model checkpoint"
    )
    parser.add_argument(
        "--single_channel",
        action="store_true",
        help="To use grayscale images"
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for evaluation"
    )
    return parser.parse_args()

if __name__ == "__main__":
    import os
    import tensorflow as tf

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    args = get_args()
    data_dir = os.path.join(args.data_path, "processed", "coco")
    
    # Initialize model
    if args.single_channel:
        model = ResNet18((224, 224, 1), mode="segmentation")
    else:
        model = ResNet18((224, 224, 3), mode="segmentation")
    
    # Load model weights
    print(f"Loading model from: {args.model_path}")
    model.load_weights(args.model_path)
    
    # Create test dataset
    print("Loading test dataset...")
    test_dataset = create_dataset_segmentation(
        data_dir,
        split="test",
        batch_size=args.batch_size,
        single_channel=args.single_channel,
    )
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, test_dataset)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Average Loss: {results['loss']:.4f}")
    print(f"Mean IoU: {results['mean_iou']:.4f}")
    print("IoU per class:")
    for i, iou in enumerate(results['class_ious']):
        print(f"  Class {i}: {iou:.4f}")

"""
Example usage:
python scripts/04_evaluate_segmentation.py --model_path models/checkpoints_seg_resnet18_new_decoder_vgg_10k/final_segmentation_model.weights.h5 --single_channel
python scripts/04_evaluate_segmentation.py --model_path models/balbi/
"""