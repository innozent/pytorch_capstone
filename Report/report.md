# PyTorch Capstone Project Report
# Cat Breed Classification System

## 1. Introduction
This project is to study on using PyTorch model to classify cat images. It implements Grad-CAM to visualized the attention of the model on one of CNN layers. ChatGPT-4o-mini is used to identify the image and given the rationale of result. Quiz is implemented to show the accuracy of each model along with human accuracy.

## 2. Dataset
- Dataset is from Kaggle from 'Geno Cat Breed Image Collection' dataset (url: https://www.kaggle.com/datasets/shawngano/gano-cat-breed-image-collection)
- Contains 15 cat breeds with 375 photos for each breed (total 5,625 photos)
- Preprocessing step
  - Resize to 256x256
  - Random Crop to 224x224
  - Random Horizontal Flip
  - Random Rotation
  - Convert to Tensor
  - Normalize with mean and std of ImageNet

## 3. Model Architecture

- Architecture of implemented models:
  - 5 Convolutional Layers 
  - 2 Fully Connected Layers
  - 1 Dropout Layer in between the two FC Layers

- Transfer Learning Models:
  - ResNet18 Model (Weights: ImageNet)
  - EfficientNetB2 Model (Weights: ImageNet)
  - VGG16 Model (Weights: ImageNet)

## 4. Training Methodology
- Training process and hyperparameters
  - Training Batch Size: 128
  - Learning Rate: 0.001
  - Momentum: 0.9
  - Epochs: 5
  - Loss Function: Cross Entropy Loss
  - Optimizer: SGD

## 5. Results and Evaluation
- Compare Training Accuracy and Loss of each model.
- Compare Training Time of each model.
- Quiz is implemented to show the accuracy of each model along with human accuracy and ChatGPT-4o-mini accuracy.

## 6. Visualization and Interpretability 
- Implemented Grad-CAM to compared and visualize each model's attention on the image
- ChatGPT-4o-mini will give the rationale of its answer.

## 7. Challenges and Solutions
- Grad-CAM target layer for each model architecture is different and need to be selected manually.
- CNN model accuracy is very low, even worse than random guess.

## 8. Future Improvements
- Apply more data augmentation techniques

## 9. Conclusion
- Summary of findings
- Final model selection and justification
- Project outcomes and achievements
