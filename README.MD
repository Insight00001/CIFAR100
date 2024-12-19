# Image classification using Tensorflow on cifar100 dataset
## Problem Statement
* The objective of this project is to develop a highly accurate image classification model using the CIFAR-100 dataset, which consists of 60,000 32x32 color images divided into 100 classes. The project aims to leverage advanced deep learning techniques to build and optimize a model that achieves superior classification accuracy and can classify any uploaded images into the defined classes.

* Objectives
1. Build an Accurate Classifier: Develop a model to classify CIFAR-100 images into one of the 100 classes.
2. Optimize Model Performance: Experiment with different architectures and hyperparameters to improve accuracy.
3. Visualize Model Insights: Create visualizations to interpret and understand model performance.
4. Deploy the Model: Build a user-friendly interface for real-time image classification.
5. Validate Hypotheses: Systematically test the hypotheses related to model performance and feature importance.

## Introduction
This repository contains the code and methodology for performing image classification using Tensorflow with the Cifar100 dataset. The Cifar100 dataset consists of a diverse set of images and is commonly used for benchmarking image classification algorithms.

## Methodology
* Data Preprocessing: the Dataset is processed to normalize and augment the images to improve model generalization
* Base Model: A basic CNN model is choosen with a simple architecture to bench mark other complex models
* Hyperparameter fine-tuning: the base model is fine-tuned on the cifar100 dataset to adapt to the specific classification task using:
* GridSearch
* Bayesian Optimization
* Transfer learning: pre-trained model (RESNET50) is used on the cifar100 dataset.
* Hypothesis: T-test is performed to determine if regular machine learning model(Logistic regression) perform better than deep learning models
* Training: the model is trained on the appropiate amount of training dataset with the appropiate optimizer and loss function
* Evaluation: the trained model performance is evaulauted on a test dataset using Accuracy, Precision Score, and recal_score
## Model Performance(RESNET50)
* Accuracy Score: 60.58%
* Precision Score: 62.14%
* Recall Score: 60.58%
## Libraies
The following Libaries were used in this project:
* Tensorflow: A framework used to impelement deep learning for training and building neural networks
* numpy: Used for Scientific computation with Python used for array manipulations and data amnipulations
* matplotlib: A plotting library for Python
* Streamlit: for creating a interactive interface for user
* Pickle: used to load the saved model
* PIL: used to load the images, resize the target image for the model

# Model Saving
the final model was saved in HDF5 format

## Usage
* Clone the respoitory: git clone https://github.com/Insight00001/CIFAR100.git
* installed the required dependencies: pip install -r requirements
* download the Cifar100 dataset: use the tensoflow in built function

## Conclusion:
This are the steps taken to perform image classification using tensorflow on Cifar100 dataset, the model was able to predict some classes correctly with a confidence score of 100%




