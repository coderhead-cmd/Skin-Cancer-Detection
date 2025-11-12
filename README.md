# Automated Diagnosis of Skin Cancer Using Deep Learning

## Overview

This repository presents a deep learning–based approach for the automated diagnosis of skin cancer using dermoscopic images. The project utilizes a ResNet50 convolutional neural network (CNN) architecture trained on the HAM10000 dataset to classify various types of skin lesions, including both benign and malignant categories.  
The goal is to develop a reliable computer-aided diagnostic (CAD) system that can assist dermatologists in early detection of skin cancer, enhancing diagnostic accuracy and reducing the workload in clinical environments.

## Abstract

Skin cancer is among the most prevalent forms of cancer worldwide, with rising incidence rates due to increased UV exposure and delayed diagnosis. Early identification of malignant lesions, such as melanoma, significantly improves patient survival rates.  This project applies transfer learning with ResNet50, a deep residual network pre-trained on ImageNet, to extract and learn intricate visual features from dermoscopic images. The model is fine-tuned to detect and classify seven distinct lesion types present in the HAM10000 dataset.  
Through comprehensive preprocessig and hyperparameter optimization, the model achieves high accuracy and demonstrates robust generalization capability. The proposed approach provides a non-invasive and cost-effective solution to assist dermatologists in clinical decision-making.

## Dataset: HAM10000

- Classes: 7 lesion types  
  - Actinic Keratoses (AKIEC)  
  - Basal Cell Carcinoma (BCC)  
  - Benign Keratosis-like Lesions (BKL)  
  - Dermatofibroma (DF)  
  - Melanocytic Nevi (NV)  
  - Vascular Lesions (VASC)  
  - Melanoma (MEL)  
- Format: JPEG images (600×450 pixels)  
- Split: 80% Training, 20% Testing  

The dataset was preprocessed for uniformity and augmented through rotation, flipping, and zooming to improve robustness and prevent overfitting.

## Methodology

1. Data Preprocessing
   - Image normalization and resizing to 224×224 pixels  
   - Data augmentation using Keras `ImageDataGenerator`  
   - Label encoding for multiclass classification  

2. Model Architecture  
   - Base model: ResNet50 (pre-trained on ImageNet) 
   - Fine-tuned final layers for 7-class classification  
   - Global Average Pooling, Dropout regularization, and Dense output layer with Softmax activation  

3. Training Configuration
   - Optimizer: Adam  
   - Loss Function: Categorical Cross-Entropy  
   - Batch Size: 32  
   - Learning Rate: 0.0001  
   - Epochs: 25–30  

4. Evaluation Metrics
   - Accuracy  
   - Precision, Recall, F1-score  
   - Confusion Matrix  
   - ROC Curve and AUC  

## Results and Discussion

The ResNet50-based CNN demonstrated high classification accuracy, effectively distinguishing between multiple lesion types. Visualizations of training curves and confusion matrices confirm that the model generalizes well without significant overfitting.

The model achieved an accuracy exceeding 90%, highlighting its potential as a dependable diagnostic support tool. Misclassifications mainly occurred between visually similar lesions such as BKL and MEL, indicating scope for further refinement using hybrid or ensemble models.

## Future Work

- Integration of explainable AI (XAI) techniques such as Grad-CAM for visual interpretability.  
- Exploration of hybrid CNN architectures and hyperparameter optimization for improved accuracy.  
- Deployment of a web or mobile-based interface for real-time clinical use.  
- Expansion to multi-modal learning, combining dermoscopic images with patient metadata.  

## Tools and Libraries

- Python 3.x  
- TensorFlow / Keras  
- NumPy, Pandas, Matplotlib, Seaborn  
- Scikit-learn  
- OpenCV  
