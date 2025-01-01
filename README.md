# ANN-Based Customer Churn prediction

This project predicts customer churn using an Artificial Neural Network (ANN). It includes data preprocessing, model training, and deployment through a Streamlit application.

---

## Table of Contents
- [Overview](#overview)  
- [Dataset](#dataset)  
- [Methodology](#methodology)  
  - [Data Collection](#data-collection)  
  - [Exploratory Data Analysis](#exploratory-data-analysis)  
  - [Data Preprocessing](#data-preprocessing)    
  - [Splitting Data](#splitting-data)  
  - [Model Building](#model-building)  
  - [Model Training](#model-training)  
  - [Model Evaluation](#model-evaluation)  
  - [Results](#results)  
- [Deployment](#deployment)  
- [Challenges and Solutions](#challenges-and-solutions)  
  - [Mixed Data Types](#mixed-data-types)  
  - [Class Imbalance](#class-imbalance)  
  - [Model Optimization](#model-optimization)  
  - [Prediction Interpretation](#prediction-interpretation)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Experiments](#experiments)  


## Overview

This project demonstrates the creation of an ANN model for binary classification. The model predicts whether a customer will leave a bank based on various features. It covers loading and preprocessing the data, encoding categorical variables, and building a multi-layer perceptron using TensorFlow Keras.

---
## Dataset
- **Name**: Churn_Modelling.csv  
- **Description**: Includes customer demographics, account details, and churn status.  
- **Key Features**: Credit score, geography, gender, age, tenure, and balance.  

---

## Methodology

### Data Collection

- Gather the data required for the project. The specified dataset contains information about customers, including features like credit score, geography, gender, age, tenure, balance, and more.

### Exploratory Data Analysis
- Removed unnecessary columns (e.g., `RowNumber`, `CustomerId`, `Surname`).  
- Encoded categorical variables and normalized numeric features

### Splitting Data  
- Divided data into training and testing sets for model training and evaluation.  

### Model Building  
- Designed an ANN using TensorFlow Keras with two hidden layers.

### Model Training  
- Trained the ANN with various hyperparameters, such as epochs and batch size.  
- Used TensorBoard to monitor accuracy and loss metrics during training.

### TensorBoard Visualization
Monitor the training process using TensorBoard for performance visualization. TensorBoard helps in visualizing the following metrics:

- **Accuracy :** Track the accuracy of the model over epochs
- **Loss** : Observe how the loss decreases over training epochs
- **Learning Rate** : Visualize how the learning rate changes during training

To set up TensorBoard, follow these steps:

  1. Import tensorBoard and create a callback:
     ```
        from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
        import datetime

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorflow_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
     ```

  2. Include the callback in the model training process:
     ```
     model.fit(X_train, y_train,
     validation_data = (X_test,y_test),
     epochs=100,
     callbacks=[tensorflow_callback, early_stopping_callback])
     ```
  3. To launch TensorBoard, use the following command in your terminal:

     ```
     %tensorboard --logdir=logs/fit
     ```
The trained model is saved as model.h5

<img width="876" alt="Screenshot 2024-12-31 at 9 52 12 PM" src="https://github.com/user-attachments/assets/23201aa3-5892-4400-88ff-624e7b43c90c" />
<img width="882" alt="Screenshot 2024-12-31 at 9 52 42 PM" src="https://github.com/user-attachments/assets/6e41ac39-3b37-4309-ba35-24b85d08727f" />
<img width="877" alt="Screenshot 2024-12-31 at 9 52 56 PM" src="https://github.com/user-attachments/assets/9d888ace-64b4-4946-9e9b-c93fcb51d0ef" />

### Model Evaluation









