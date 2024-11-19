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
        rle['counts'] = rle['counts'].decode('utf-8')
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
                self.annotations.append({
                    'id': self.ann_id,
                    'image_id': self.img_id,
                    'category_id': class_idx + 1,  # COCO uses 1-based indexing
                    'segmentation': self.encode_binary_mask(gt_mask),
                    'area': float(gt_mask.sum()),
                    'bbox': maskUtils.toBbox(self.encode_binary_mask(gt_mask)).tolist(),
                    'iscrowd': 0
                })
                self.ann_id += 1
        
        # Process predictions
        for class_idx in range(self.num_classes):
            
            pred_scores = y_pred[..., class_idx]
            pred_mask = (pred_scores > pred_threshold).astype(np.uint8)
            if pred_mask.sum() > 0:  # If there's a prediction
                confidence = float(pred_scores[pred_mask > 0].mean())
                self.predictions.append({
                    'image_id': self.img_id,
                    'category_id': class_idx + 1,  # COCO uses 1-based indexing
                    'segmentation': self.encode_binary_mask(pred_mask),
                    'score': confidence,
                    'bbox': maskUtils.toBbox(self.encode_binary_mask(pred_mask)).tolist(),
                    'area': float(pred_mask.sum())
                })
        
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
            'images': [{'id': i, 'height': 0, 'width': 0} for i in range(self.img_id)],
            'categories': [{'id': cat_id, 'name': str(cat_id)} for cat_id in self.selected_cat_ids],
            'annotations': self.annotations
        }
        coco_gt.createIndex()
        
        # Create COCO predictions dataset
        coco_dt = coco_gt.loadRes(self.predictions)
        
        # Create COCO evaluator
        coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
        
        # Run evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract metrics
        metrics = {
            'mAP': coco_eval.stats[0],  # AP at IoU=0.50:0.95
            'mAP_50': coco_eval.stats[1],  # AP at IoU=0.50
            'mAP_75': coco_eval.stats[2],  # AP at IoU=0.75
            'mAP_small': coco_eval.stats[3],  # AP for small objects
            'mAP_medium': coco_eval.stats[4],  # AP for medium objects
            'mAP_large': coco_eval.stats[5],  # AP for large objects
            'AR_max_1': coco_eval.stats[6],  # AR given 1 detection per image
            'AR_max_10': coco_eval.stats[7],  # AR given 10 detections per image
            'AR_max_100': coco_eval.stats[8],  # AR given 100 detections per image
            'AR_small': coco_eval.stats[9],  # AR for small objects
            'AR_medium': coco_eval.stats[10],  # AR for medium objects
            'AR_large': coco_eval.stats[11]  # AR for large objects
        }
        
        return metrics

def example_usage(processed_data_dir, split="val", num_samples=50):
    """
    Test the COCO segmentation evaluator using preprocessed data and random predictions.

    Args:
        processed_data_dir (str): Path to the processed data directory
        split (str): Data split to use ('train' or 'val')
        num_samples (int): Number of samples to evaluate
    """
    # Define the selected category IDs based on the list of classes to evaluate
    data_dir = os.path.join("/mnt/f/ssl_images/data", "coco")
    annotations_path = os.path.join(data_dir, "annotations", f"instances_{split}2017.json")
    coco = COCO(annotations_path)
    selected_cat_ids = coco.getCatIds(
        catNms=[
            "person",
            "car",
            "chair",
            "book",
            "bottle",
            "cup",
            "dining table",
            "traffic light",
            "bowl",
            "handbag",
        ]
    )
    
    # Initialize evaluator with the selected categories
    evaluator = COCOSegmentationEvaluator(num_classes=10, selected_cat_ids=selected_cat_ids)

    # Construct path to the segmentation data
    segmentation_dir = os.path.join(
        processed_data_dir, "coco", "segmentation", f"{split}2017"
    )

    # Get list of all mask files
    mask_files = glob(os.path.join(segmentation_dir, "mask_*.npy"))[:num_samples]

    if not mask_files:
        raise ValueError(f"No mask files found in {segmentation_dir}")

    # Process each sample
    for mask_file in mask_files:
        # Load ground truth mask
        y_true = np.load(mask_file)

        # Skip if y_true is all zeros
        if np.sum(y_true) == 0:
            continue

        # Generate random predictions with same shape as ground truth
        y_pred = y_true

        # Update evaluator with this sample
        evaluator.update(y_true, y_pred)


    
    # Get metrics
    metrics = evaluator.get_metrics()
    
    # Print results
    print("\nCOCO Evaluation Results:")
    print("-" * 50)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    processed_data_dir = "/mnt/f/ssl_images/data/processed"
    example_usage(processed_data_dir, split="val", num_samples=50)
