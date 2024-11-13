import argparse
import cv2
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import random
import sys

from typing import List, Tuple
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.resnet import ResNet18


os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '1'

def generateRandomMask(
        mask_size: Tuple[int, int],
        img_size: Tuple[int, int],
        n_masks: int
        ):

    mask = np.full((img_size[0], img_size[1]), 255)

    for _ in range(n_masks):
        x = random.randint(0, img_size[0]- mask_size[0])
        y = random.randint(0, img_size[1]- mask_size[1])
        mask[x:x+mask_size[0], y:y+mask_size[1]] = 0

    return mask

def get_dataset(
        dataset_path: str,
        ) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:

    x_test, y_true, img_name = [], [], []
    for img in os.listdir(dataset_path):
        # Read images
        img_path = os.path.join(dataset_path, img)
        gt_image = cv2.imread(img_path)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        # Resize and generate input image
        gt_image = cv2.resize(gt_image, (224, 224))
        # Generate masked image
        mask = generateRandomMask((32, 32), (224, 224), 5)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        input_image = (gt_image * (mask / 255))
        # Insert batch axis
        input_image = tf.expand_dims(tf.cast(input_image, tf.float32) / 255., axis=0)
        gt_image = tf.expand_dims(tf.cast(gt_image, tf.float32) / 255., axis=-1)

        x_test.append(input_image)
        y_true.append(gt_image)
        img_name.append(img.split('.')[0])

    return x_test, y_true, img_name

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_dir', type=str, help="Test set folder path.")
    parser.add_argument('-w', '--weights', type=str, help="Model weights path")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    output_path = "inference"
    os.makedirs(output_path, exist_ok=True)
    
    x_test, y_true, img_name = get_dataset(args.test_dir)

    model = ResNet18(mode='inpainting')
    model.summary()

    model.load_weights(args.weights)

    for (input_img, gt_image, name) in tqdm(zip(x_test, y_true, img_name), total=len(x_test)):
        predict = model.predict(input_img)
        input_img = tf.squeeze(input_img, axis=0)*255
        gt_img = tf.squeeze(gt_image)*255
        predict = tf.squeeze(predict, axis=0)*255
        
        img_path = os.path.join(output_path, name)
        os.makedirs(img_path, exist_ok=True)
        tf.keras.preprocessing.image.save_img(os.path.join(img_path, "input_image.png"), input_img)
        tf.keras.preprocessing.image.save_img(os.path.join(img_path, "ground_truth.jpg"), gt_img)
        tf.keras.preprocessing.image.save_img(os.path.join(img_path, "prediction.png"), predict)
        
