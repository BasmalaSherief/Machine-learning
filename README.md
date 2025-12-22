# Machine Learning Assignments

This repository contains implementations of various Machine Learning algorithms and deep learning models applied to different datasets. The projects cover classification, regression, and unsupervised learning tasks.

## Project Structure

### 1. Autoencoder
An unsupervised learning project using Neural Networks for dimensionality reduction (image embedding) and image restoration on the MNIST dataset.

### 2. KNN Classifier
Implementation of the K-Nearest Neighbors algorithm applied to two different domains:
* **MNIST Dataset:** Handwritten digit classification with performance analysis across different `k` values.
* **Wine Dataset:** Classification of wine varieties including data normalization and PCA visualization.

### 3. Naive Bayes Classifier
Implementation of the Categorical Naive Bayes algorithm for prediction tasks:
* **Breast Cancer Prediction:** Predicting cancer recurrence events.
* **Weather Prediction:** Predicting the decision to play sports based on weather conditions.

### 4. Neural Network Regression
A regression task estimating the health status (Compressor and Turbine decay) of Naval Propulsion Plants. Includes workflows for:
* Data preprocessing and normalization.
* Model training with multi-start optimization.
* Hyperparameter tuning using Cross-Validation.

## Requirements
* Python 3.x
* TensorFlow / Keras
* Scikit-learn
* Pandas
* NumPy
* Matplotlib