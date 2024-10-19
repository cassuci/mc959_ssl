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
├── notebooks/
├── scripts/
│   ├── 00_download_data.py
│   ├── 01_data_preparation.py
│   ├── 02_pretext_task_training.py
│   ├── 03_supervised_finetuning.py
│   └── 04_evaluation.py
├── src/
│   ├── libs/
│   │   ├── data_processing.py
│   │   └── visualization.py
│   ├── models/
│   │   ├── base_model.py
│   │   └── resnet.py
│   └── utils/
│       └── metrics.py
├── tests/
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/self_supervised_learning_project.git
   cd self_supervised_learning_project
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
   python scripts/01_data_preparation.py
   ```
   This script preprocesses the data for both pretext and downstream tasks.

3. Train on pretext tasks:
   ```
   python scripts/02_pretext_task_training.py
   ```
   This trains the model on inpainting and colorization tasks.

4. Fine-tune on downstream task:
   ```
   python scripts/03_supervised_finetuning.py
   ```
   This fine-tunes the pre-trained model on the classification task.

5. Evaluate the model:
   ```
   python scripts/04_evaluation.py
   ```
   This evaluates the fine-tuned model and generates performance metrics.

## Model Architecture

The project uses a ResNet18 architecture for both pretext and downstream tasks. The model is implemented in `src/models/resnet.py` and extends the `BaseModel` class defined in `src/models/base_model.py`.

## Pretext Tasks

1. Inpainting: The model learns to reconstruct masked portions of images.
2. Colorization: The model learns to colorize grayscale images.

## Downstream Task

Image Classification: The pre-trained model is fine-tuned for multi-class image classification using the Pascal VOC dataset.

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