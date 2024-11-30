# Self-Supervised Learning Project

This project explores self-supervised learning methods for image classification and segmentation tasks using the COCO and Pascal VOC datasets. It implements a pipeline that includes pretext task training (inpainting and colorization) followed by supervised fine-tuning for downstream tasks.

## Project Structure

```
self_supervised_learning_project/
├── data/
│   ├── coco/
│   ├── pascal_voc/
│   └── processed/
├── models/
├── scripts/
│   ├── 00_download_data.py
│   ├── 01_data_preparation_coco.py
│   ├── 01_data_preparation_pascalvoc.py
│   ├── 02_coloring_task_training.py
│   ├── 02_inpainting_task_training.py
│   ├── 03_classification_task_training.py
│   ├── 03_segmentation_task_training.py
│   └── 04_evaluate_segmentation.py
├── src/
│   ├── libs/
│   │   ├── data_loading_pascal.py
│   │   ├── data_loading.py
│   │   ├── data_processing.py
│   │   └── eval_segmentation.py
│   │   └── visualization.py
│   ├── models/
│   │   └── resnet.py
│   └── utils/
│       └── metrics.py
│       └── evaluation.py
├── tests/
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/cassuci/mc959_ssl.git
   cd mc959_ssl
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Follow these steps to run the project:

1. Download the datasets:
   ```
   python scripts/00_download_data.py
   ```
   This will download the COCO and Pascal VOC datasets to the `data/` directory.

2. Prepare the data:
   ```
   python scripts/01_data_preparation_coco.py
   ```
   ```
   python scripts/01_data_preparation_pascalvoc.py
   ```
   This script preprocesses the data for both pretext and downstream tasks.

3. Train the baseline models:
   ```
   python scripts/03_classification_task_training.py
   ```
   ```
   python scripts/03_segmentation_task_training.py
   ```
   This trains the model on the classification and segmentation tasks, without pretrained weights.

4. Train on pretext tasks:
   ```
   python scripts/02_inpainting_task_training.py
   ```
   ```
   python scripts/02_coloring_task_training.py
   ```
   This trains the model on inpainting and colorization tasks.

5. Fine-tune on downstream task:
   ```
   python scripts/03_classification_task_training.py --pretrained_model <path to model checkpoint>
   ```
   ```
   python scripts/03_segmentation_task_training.py --pretrained_model <path to model checkpoint>
   ```
   This fine-tunes the pre-trained model on the classification and segmentation tasks.

6. Evaluate the model:
   ```
   python scripts/04_evaluate_segmenation.py --model_path <path to model checkpoint>
   ```
   This evaluates the fine-tuned model and generates performance metrics.

## Model Architecture

The project uses a ResNet18 architecture for both pretext and downstream tasks. The model is implemented in `src/models/resnet.py`.

## Pretext Tasks

1. Inpainting: The model learns to reconstruct masked portions of images.
2. Colorization: The model learns to colorize grayscale images.

## Downstream Task

1. Image Classification: The model is trained for binary image classification using the Pascal VOC dataset, for person detection.
2. Image segmentation: The model is trained to segment objects from three classes (person, car and chair).

## Results

After running the evaluation script, you'll find the performance metrics printed in the console. These include accuracy, precision, recall, F1-score, and AUC-ROC for the classification task.

## Visualization

The project includes visualization tools in `src/libs/visualization.py`. These are used to plot training histories, feature maps, and to visualize the latent space of the trained model.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- COCO dataset: https://cocodataset.org/
- Pascal VOC dataset: http://host.robots.ox.ac.uk/pascal/VOC/

## Contact

For any questions or feedback, please open an issue in the GitHub repository.