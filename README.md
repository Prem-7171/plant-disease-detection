# Plant Disease Detection using CNN

A deep learning-based computer vision project for classifying plant leaf images into 38 different disease and healthy-leaf categories using a custom Convolutional Neural Network (CNN) built with TensorFlow/Keras.

## Overview

This project explores image classification for plant disease detection using the PlantVillage dataset.

The complete workflow includes:

- Dataset exploration and class distribution analysis
- Image inspection and size analysis
- Image preprocessing and normalization
- Training/validation dataset preparation
- Class-weight calculation for imbalanced classes
- Custom CNN architecture design
- Model training with GPU acceleration
- Learning-rate scheduling
- Early stopping and model checkpointing
- Training performance analysis

## Dataset

The project uses the **PlantVillage dataset** containing **54,305 images across 38 classes**.

The dataset contains healthy and diseased leaf images from multiple crops, including:

- Apple
- Blueberry
- Cherry
- Corn
- Grape
- Orange
- Peach
- Pepper
- Potato
- Raspberry
- Soybean
- Squash
- Strawberry
- Tomato

The dataset itself is **not included in this repository** because of its size.

## Data Distribution

The dataset contains 38 classes with varying numbers of images.

The project performs class-distribution analysis before training and calculates class weights to reduce the impact of class imbalance during model training.

The dataset contains:

- Total images: **54,305**
- Training images: **43,456**
- Validation images: **10,849**
- Validation split: **20%**

The largest class contains 5,507 images while the smallest contains 152 images.

## Model Architecture

A custom CNN architecture was developed using TensorFlow/Keras.

### Architecture

```text
Input: 128 × 128 × 3
        │
        ▼
Conv2D (32 filters) + BatchNorm + MaxPooling
        │
        ▼
Conv2D (64 filters) + BatchNorm + MaxPooling
        │
        ▼
Conv2D (128 filters) + BatchNorm + MaxPooling
        │
        ▼
Conv2D (256 filters) + BatchNorm + MaxPooling
        │
        ▼
Flatten
        │
        ▼
Dense (256) + Dropout (0.4)
        │
        ▼
Dense (128) + Dropout (0.3)
        │
        ▼
Dense (38) + Softmax
```

### Model Statistics

- Input size: `128 × 128 × 3`
- Output classes: `38`
- Total parameters: **4,622,694**
- Trainable parameters: **4,621,734**

## Training

The model was trained using:

- Optimizer: **Adam**
- Initial learning rate: **0.001**
- Loss: **Categorical Crossentropy**
- Metric: **Accuracy**
- Batch size: **32**
- Maximum epochs: **30**

The training pipeline also uses:

- `ModelCheckpoint`
- `EarlyStopping`
- `ReduceLROnPlateau`
- Class weights

The best model is automatically saved based on validation accuracy.

## Results

The model achieved a best validation accuracy of:

**97.99%**

The best validation accuracy was achieved during the later stages of training after learning-rate reduction.

The saved model is available as:

```text
models/best_model.keras
```

## Training Performance

Training and validation accuracy/loss curves are available in the `results/` directory.

![Training History](results/history_Custom_CNN.png)

## Hardware

The model was trained using GPU acceleration with an NVIDIA RTX 4050 Laptop GPU.

## Project Structure

```text
plant-disease-detection/
│
├── notebooks/
│   └── plant_disease_detection.ipynb
│
├── models/
│   └── best_model.keras
│
├── results/
│   └── history_Custom_CNN.png
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Prem-7171/plant-disease-detection.git
cd plant-disease-detection
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Dataset Setup

Download the PlantVillage dataset separately and configure the dataset path in the notebook.

The original notebook uses a local dataset path, so this path must be changed to match your local environment before running the notebook.

## Running the Project

Open:

```text
notebooks/plant_disease_detection.ipynb
```

Run the notebook sequentially to reproduce the data exploration, preprocessing, model construction, training, and evaluation workflow.

## Future Improvements

Potential improvements include:

- Transfer learning using pretrained vision models
- Data augmentation
- More rigorous class-imbalance analysis
- External test-set evaluation
- Robustness testing under different lighting and image conditions
- Domain-shift evaluation
- Model explainability using Grad-CAM
- Deployment as an inference API or web application
- Experiment tracking and reproducibility improvements

## Author

**Prem Palkar**

B.Tech Computer Science & Engineering — AI & ML

GitHub: [Prem-7171](https://github.com/Prem-7171)