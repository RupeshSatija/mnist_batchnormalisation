# MNIST Digit Recognition with PyTorch

A CNN-based solution for MNIST digit recognition achieving 99.4%+ validation accuracy.

## Model Architecture

The model uses a modern CNN architecture with:
- Convolutional layers with batch normalization
- Dropout for regularization
- Global Average Pooling (GAP) instead of Fully Connected layers
- Under 20,000 parameters

### Key Features:
- BatchNorm after each conv layer
- Dropout (15%) for regularization
- GAP for final feature aggregation
- Data augmentation with controlled rotations

## Performance Requirements
- [x] Validation accuracy > 99.4%
- [x] Parameter count < 20,000
- [x] Uses Batch Normalization
- [x] Uses Dropout
- [x] Uses Global Average Pooling

## Training Details
- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 64
- Epochs: 20
- LR Scheduler: StepLR (step_size=5, gamma=0.5)

## Data Augmentation
- Rotation: ±7 degrees
- Translation: 10%
- Scale: ±10%
- Shear: ±10 degrees
- Random Perspective: 20%
- Random Erasing: 20%

## Project Structure 
.
├── README.md
├── requirements.txt
├── test_model.py
└── train.py