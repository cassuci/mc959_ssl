# src/utils/metrics.py

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def intersection_over_union(y_true, y_pred):
    """Calculate Intersection over Union (IoU) for segmentation tasks."""
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def dice_coefficient(y_true, y_pred):
    """Calculate Dice coefficient for segmentation tasks."""
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + 1.) / (np.sum(y_true) + np.sum(y_pred) + 1.)

def pixel_accuracy(y_true, y_pred):
    """Calculate pixel-wise accuracy for segmentation tasks."""
    correct_pixels = np.sum(y_true == y_pred)
    total_pixels = y_true.size
    return correct_pixels / total_pixels

def classification_metrics(y_true, y_pred, y_prob=None):
    """Calculate various classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    
    if y_prob is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
    
    return metrics

class SegmentationMetrics(tf.keras.metrics.Metric):
    """Custom Keras metric for segmentation tasks."""
    def __init__(self, name='segmentation_metrics', **kwargs):
        super().__init__(name=name, **kwargs)
        self.iou = self.add_weight(name='iou', initializer='zeros')
        self.dice = self.add_weight(name='dice', initializer='zeros')
        self.pixel_acc = self.add_weight(name='pixel_acc', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred > 0.5, tf.bool)

        self.iou.assign_add(tf.py_function(intersection_over_union, [y_true, y_pred], tf.float32))
        self.dice.assign_add(tf.py_function(dice_coefficient, [y_true, y_pred], tf.float32))
        self.pixel_acc.assign_add(tf.py_function(pixel_accuracy, [y_true, y_pred], tf.float32))
        self.count.assign_add(1)

    def result(self):
        return {
            'iou': self.iou / self.count,
            'dice': self.dice / self.count,
            'pixel_accuracy': self.pixel_acc / self.count
        }

    def reset_state(self):
        self.iou.assign(0)
        self.dice.assign(0)
        self.pixel_acc.assign(0)
        self.count.assign(0)