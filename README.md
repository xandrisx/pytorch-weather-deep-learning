# Weather Image Classification using Deep Learning

## 🌟 Project Overview
This deep learning project implements a weather condition classifier using transfer learning with ResNet18 architecture.
The system can accurately classify weather images into four distinct categories: cloudy, rainy, sunny, and sunrise conditions.

## 🎯 Problem Statement
Weather classification from images has numerous practical applications, from meteorological studies to automated weather reporting systems.
This project demonstrates how deep learning can be effectively used for automated weather condition recognition.

## 📊 Dataset
The project uses a curated weather image dataset containing four categories:

    Cloudy conditions
    Rainy conditions
    Sunny conditions
    Sunrise scenes

## Dataset Structure:

weather_images/
    ├── cloudy/
    ├── rainy/
    ├── sunny/
    └── sunrise/
    
### Why This Dataset?
This dataset is particularly suitable because it:

    -Provides balanced representation of different weather conditions
    -Contains high-quality, real-world images
    -Offers sufficient variety for robust model training
    -Includes distinct visual patterns for each weather category

## 🤖 AI Model Architecture
### Model Details

    -Base Architecture: ResNet18 (pretrained on ImageNet)
    -Transfer Learning: Utilized pretrained weights for feature extraction
    -Output Layer: Modified for 4-class classification
    -Input Size: 224x224 pixels, 3 channels

### Why ResNet18?

    -Proven architecture for image classification tasks
    -Efficient training with residual connections
    -Good balance between model complexity and performance
    -Excellent feature extraction capabilities

## 🛠️ Implementation Details
### Training Parameters

    -Training/Test Split: 80/20
    -Batch Size: 32
    -Optimizer: Adam (lr=0.001)
    -Loss Function: CrossEntropyLoss
    -Epochs: 20

### Data Preprocessing

    -Resize to 224x224 pixels
    -Tensor conversion
    -Normalization using ImageNet statistics

## 📝 Installation & Usage
### Option 1: Google Colab (Recommended)

    -Open the provided Colab notebook: 
    -Mount your Google Drive
    -Upload the dataset to your Drive
    -Run all cells sequentially
Option 2: Local Installation

python
# Clone repository
git clone [repository-url]

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py

## 📊 Results & Evaluation

    -Model achieves significant accuracy in weather classification
    -Confusion matrix visualization provided
    -Per-class accuracy metrics available
    -Real-time inference capabilities


## 🔧 Future Improvements

    -Data augmentation implementation
    -Architecture experimentation
    -Hyperparameter optimization
    -Ensemble methods integration

## 📚 Sources

    -Weather Dataset: https://data.mendeley.com/datasets/4drtyfjtfy/1
    -PyTorch Documentation: https://pytorch.org/docs/stable/index.html
    -Deep Learning Research: https://www.ibm.com/topics/deep-learning

## 👤 Author
### Pál András Richárd

    -Email: palandrasrichard@gmail.com
    -Neptun: CDNNLO
    -Institution: Budapest University of Technology and Economics

## 📄 License
### This project is licensed under the MIT License - see the LICENSE file for details Note: This project was developed as part of the PythonAI course at BME.
