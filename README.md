# Heart Disease Prediction Using K-Nearest Neighbors (KNN)

This repository contains a Jupyter Notebook that uses the **K-Nearest Neighbors (KNN)** algorithm to predict the risk of heart disease based on medical features. The dataset contains various health-related attributes such as age, cholesterol levels, blood pressure, and others, which are used to classify whether a person is at risk of heart disease or not.

## Project Overview:
- **Goal**: Predict whether a patient is at risk of heart disease based on health features.
- **Dataset**: The dataset used for this project contains several medical features such as age, sex, cholesterol, etc., and a binary target indicating whether a person is at risk of heart disease.
- **Algorithm**: The **K-Nearest Neighbors (KNN)** algorithm is used to make predictions.
- **Model Evaluation**: The model is evaluated using classification metrics like accuracy, precision, recall, and F1 score.

## Running the Notebook

### Option 1: Run in Google Colab
To easily run the notebook in Google Colab without setting up a local environment, click the link below:
[Open in Google Colab](https://colab.research.google.com/github/HaardJoshi/heart_disease_prediction/blob/main/model.ipynb)

### Option 2: Run on Your Local Machine
To run the notebook locally, follow these steps:

1. **Clone this repository** to your local machine:
   ```bash
   git clone https://github.com/HaardJoshi/heart-disease-prediction-knn.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd heart-disease-prediction-knn
   ```

3. **Install dependencies**:
   - Make sure you have Python 3.x installed.
   - Install the required Python libraries using `pip`:
     ```bash
     pip install -r requirements.txt
     ```

4. **Run the notebook**:
   - Open the notebook in Jupyter Notebook:
     ```bash
     jupyter notebook
     ```

5. **Access the notebook** in your browser, and open the `model.ipynb` file.

## Description of the Notebook

The notebook `model.ipynb` includes the following steps:

1. **Data Exploration**: We start by loading the dataset, checking its basic structure, and visualizing the data.
2. **Data Preprocessing**: The dataset is cleaned, and necessary transformations are applied (e.g., scaling the features).
3. **Model Training**: The K-Nearest Neighbors (KNN) algorithm is trained on the dataset to classify the risk of heart disease.
4. **Model Evaluation**: The model's performance is evaluated using accuracy, confusion matrix, and other classification metrics.
5. **Hyperparameter Tuning**: The notebook also explores finding the best number of neighbors for the KNN model.
