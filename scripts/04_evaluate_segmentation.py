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

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import os
from glob import glob


class COCOSegmentationEvaluator:
    """
    Evaluator class for COCO instance segmentation tasks using official pycocotools.
    """

    def __init__(self, num_classes, selected_cat_ids):
        """
        Initialize the evaluator.

        Args:
            num_classes (int): Number of classes in the segmentation task
            selected_cat_ids (list): List of category IDs to evaluate
        """
        self.num_classes = num_classes
        self.selected_cat_ids = selected_cat_ids
        self.reset()

    def reset(self):
        """Reset all evaluation metrics."""
        self.annotations = []
        self.predictions = []
        self.img_id = 0
        self.ann_id = 0

    def encode_binary_mask(self, mask):
        """
        Encode binary mask to RLE format used by COCO.

        Args:
            mask (np.ndarray): Binary mask of shape (H, W)

        Returns:
            dict: RLE encoded mask
        """
        rle = maskUtils.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    def update(self, y_true, y_pred, pred_threshold=0.5):
        """
        Update metrics with a new batch of predictions.

        Args:
            y_true (np.ndarray): Ground truth masks of shape (H, W, num_classes)
            y_pred (np.ndarray): Predicted masks of shape (H, W, num_classes) with confidence scores
            pred_threshold (float): Threshold for converting prediction probabilities to binary masks
        """
        height, width = y_true.shape[:2]

        # Process ground truth
        for class_idx in range(self.num_classes):
            gt_mask = y_true[..., class_idx]
            if gt_mask.sum() > 0:  # If there's an instance
                self.annotations.append(
                    {
                        "id": self.ann_id,
                        "image_id": self.img_id,
                        "category_id": class_idx + 1,  # COCO uses 1-based indexing
                        "segmentation": self.encode_binary_mask(gt_mask),
                        "area": float(gt_mask.sum()),
                        "bbox": maskUtils.toBbox(self.encode_binary_mask(gt_mask)).tolist(),
                        "iscrowd": 0,
                    }
                )
                self.ann_id += 1

        # Process predictions
        for class_idx in range(self.num_classes):
            pred_scores = y_pred[..., class_idx]
            pred_mask = (pred_scores > pred_threshold).astype(np.uint8)
            if pred_mask.sum() > 0:  # If there's a prediction
                confidence = float(pred_scores[pred_mask > 0].mean())
                self.predictions.append(
                    {
                        "image_id": self.img_id,
                        "category_id": class_idx + 1,  # COCO uses 1-based indexing
                        "segmentation": self.encode_binary_mask(pred_mask),
                        "score": confidence,
                        "bbox": maskUtils.toBbox(self.encode_binary_mask(pred_mask)).tolist(),
                        "area": float(pred_mask.sum()),
                    }
                )

        self.img_id += 1

    def get_metrics(self):
        """
        Compute final evaluation metrics using COCO evaluation protocol.

        Returns:
            dict: Dictionary containing various evaluation metrics
        """
        # Create COCO ground truth dataset
        coco_gt = COCO()
        coco_gt.dataset = {
            "images": [{"id": i, "height": 0, "width": 0} for i in range(self.img_id)],
            "categories": [
                {"id": cat_id, "name": str(cat_id)} for cat_id in self.selected_cat_ids
            ],
            "annotations": self.annotations,
        }
        coco_gt.createIndex()

        # Create COCO predictions dataset
        coco_dt = coco_gt.loadRes(self.predictions)

        # Create COCO evaluator
        coco_eval = COCOeval(coco_gt, coco_dt, "segm")

        # Run evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics
        metrics = {
            "mAP": coco_eval.stats[0],  # AP at IoU=0.50:0.95
            "mAP_50": coco_eval.stats[1],  # AP at IoU=0.50
        }

        return metrics



def iou_metric(
    y_true: tf.Tensor, y_pred: tf.Tensor, num_classes: int = 3
) -> tf.Tensor:
    # Binarize predictions
    y_pred_bin = tf.cast(y_pred, tf.float32)

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

def evaluate_model(model, test_dataset, evaluator, num_classes=3):
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
        
        # Get the index of the highest value in the last axis (channels)
        max_channel_indices = np.argmax(y_pred, axis=-1)  # Shape: (batch, height, width)

        # Create a one-hot encoded mask
        y_pred_argmax = np.zeros_like(y_pred, dtype=np.uint8)  # Shape: (batch, height, width, channels)

        # Create meshgrid for all dimensions
        batch_size, height, width = max_channel_indices.shape
        batch_idx, row_idx, col_idx = np.meshgrid(np.arange(batch_size),
                                                np.arange(height),
                                                np.arange(width),
                                                indexing='ij')

        # Set the appropriate indices to 1
        y_pred_argmax[batch_idx, row_idx, col_idx, max_channel_indices] = 1
        
        # Compute IoU for each class
        batch_ious = iou_metric(y_batch, y_pred_argmax, num_classes=num_classes)
        class_ious += np.array(batch_ious)

        for j in range(x_batch.shape[0]):
            # Load ground truth mask
            y_i = y_batch[j].numpy().astype(np.uint8)  # Cast y_true to uint8

            y_pred_i = y_pred[j]

            # Skip if y_true is all zeros
            if np.sum(y_i[:, :, :10]) == 0:
                continue

            # Update evaluator with this sample
            evaluator.update(y_i, y_pred_i)
            
        num_batches += 1
    
    metrics = evaluator.get_metrics()

    # Compute averages
    avg_loss = total_loss / num_batches
    avg_class_ious = class_ious / num_batches
    mean_iou = np.mean(avg_class_ious)
    
    return {
        'loss': float(avg_loss),
        'mean_iou': float(mean_iou),
        'coco_map': metrics['mAP'],
        'coco_map50': metrics['mAP_50'],
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

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

    
    # Define the selected category IDs based on the list of classes to evaluate
    annotations_path = os.path.join(args.data_path, 'coco', "annotations", f"instances_val2017.json")
    coco = COCO(annotations_path)
    selected_cat_ids = coco.getCatIds(
        catNms=[
            "person",
            "car",
            "chair",
        ]
    )
    
    # Initialize evaluator with the selected categories
    evaluator = COCOSegmentationEvaluator(num_classes=3, selected_cat_ids=selected_cat_ids)


    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, test_dataset, evaluator)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Average Loss: {results['loss']:.4f}")
    print(f"Mean IoU: {results['mean_iou']:.4f}")
    print(f"COCO mAP: {results['coco_map']:.4f}")
    print(f"COCO mAP@IoU50: {results['coco_map50']:.4f}")
    print("IoU per class:")
    for i, iou in enumerate(results['class_ious']):
        print(f"  Class {i}: {iou:.4f}")

"""
Example usage:
python scripts/04_evaluate_segmentation.py --model_path models/checkpoints_seg_resnet18_new_decoder_vgg_10k/final_segmentation_model.weights.h5 --single_channel
python scripts/04_evaluate_segmentation.py --model_path tf218_color_seg.keras --single_channel
python scripts/04_evaluate_segmentation.py --model_path models/balbi/best_segmentation_model.keras
"""