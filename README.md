# 🚗 Vehicle Classification System

A deep learning-based vehicle classification system built using EfficientNet and Streamlit. This project classifies uploaded vehicle images into predefined vehicle categories using a trained convolutional neural network model.

The system provides an interactive web application where users can upload vehicle images and receive real-time classification predictions with confidence scores.

### 📌 Project Overview

Vehicle classification is an important task in computer vision that can be applied in:

Smart traffic systems
Automated toll systems
Parking management
Transportation analytics
Surveillance systems

This project uses transfer learning with EfficientNet to classify vehicle images accurately while maintaining efficient performance.

The system uses:

EfficientNetB0
Deep Learning
Transfer Learning
Image Classification
TensorFlow/Keras
Streamlit Interactive Application

### ⚠️ Note:
Running the training notebook may take a long time due to:
- Large dataset size
- Image preprocessing operations
- Deep learning model training
- GPU/CPU processing requirements

Training duration depends heavily on your hardware specifications.

### 🚀 Features
Upload and classify vehicle images
Real-time prediction results
Confidence score display
EfficientNet-based image classification
Interactive Streamlit web application
Lightweight and easy-to-use interface

### 🛠️ Technologies Used
Python
Streamlit
TensorFlow
Keras
NumPy
Pillow (PIL)
EfficientNetB0
Deep Learning
Transfer Learning

### 📂 Dataset

Dataset used:
MY-VID: A Malaysian Vehicle Image Dataset for Intelligent Transportation System and Road Safety.

Dataset link:
https://data.mendeley.com/datasets/n77t4fn8p8/1

The dataset contains:
- 8,832 vehicle images
- 47,975 vehicle annotations
- Multiple vehicle categories
- Malaysian traffic environment images
- Various lighting and weather conditions

### ⚙️ Project Workflow
- Dataset Preparation
- Image Preprocessing
- Model Training using EfficientNetB0
- Model Evaluation
- Model Saving
- Streamlit Application Development
- Real-Time Vehicle Classification

### 📁 Project Structure
- app.py
- data_preparation.ipynb
- README.md

### ▶️ How to Run
1. Run Data Preparation First

Before launching the application, make sure to run:
```
data_preparation.ipynb
```
This notebook performs:

Data preprocessing
Image preparation
Model training
Model evaluation
Model saving

⚠️ Training may take a significant amount of time because of the large dataset size and image processing operations.

2. Run the Streamlit Application
After completing model training, run:
```
streamlit run app.py
```

### 🧠 Model Architecture

This project uses:

EfficientNetB0
Transfer Learning
Binary Classification using Sigmoid Output

The application:

Loads the trained .keras model
Preprocesses uploaded images
Predicts vehicle classes
Displays confidence scores

### 📌 Key Highlights
- Deep learning-based image classificatio
- Transfer learning with EfficientNetB0
- Interactive deployment using Streamlit
- Real-time image prediction
- Lightweight and responsive application

### 👨‍💻 Authors
- Muhammad Afiq bin Muhd Azri Fahmi
- Fadzlul Baqi Faez bin Fadzil
- Anukthai Seeyakmani A/L Kuson
- Muhammad Kamil Zaki bin Iskandar Al-Thani
