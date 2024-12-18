# Heart Disease Risk Prediction Using KNN

![Project Status](https://img.shields.io/badge/Status-Completed-green)

## Overview
This project implements a **K-Nearest Neighbors (KNN)** model to predict the risk of heart disease using a structured dataset. The workflow adheres to the standard Data Science Lifecycle, including data preprocessing, model training, hyperparameter optimization, and evaluation.

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Dataset Overview](#dataset-overview)
3. [Project Workflow](#project-workflow)
4. [Setup and Installation](#setup-and-installation)
5. [Results](#results)
6. [Key Features](#key-features)
7. [Future Improvements](#future-improvements)
8. [Contributors](#contributors)

---

## Problem Statement
Predict whether an individual is at risk of heart disease based on medical and demographic features.

## Dataset Overview
- **Source**: `data-heart.csv`
- **Size**: 303 samples, 14 features.
- **Target Variable**: `target` (1 = at risk, 0 = no risk).
- **Features**: Age, sex, cholesterol, blood pressure, etc.

## Project Workflow
1. **Data Preparation**:
   - Load the dataset.
   - Scale the features using `StandardScaler`.
2. **Modeling**:
   - Train a KNN classifier.
   - Optimize hyperparameters (`n_neighbors`, `random_state`) using grid search.
3. **Evaluation**:
   - Compute metrics: Accuracy, Precision, Recall, F1-score.
   - Visualize results with confusion matrices and accuracy plots.

## Setup and Installation

### Prerequisites
- Python 3.8+
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `plotly` (optional for advanced visualizations)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/heart-disease-knn.git
   cd heart-disease-knn
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook heart_disease_knn.ipynb
   ```

## Results
- **Best Accuracy**: 98.36%
- **Optimal Hyperparameters**:
  - `n_neighbors`: 19
  - `random_state`: 72
- **Confusion Matrix**:
  ```
  [[27, 3],
   [ 3, 28]]
  ```

## Key Features
- Implements a full pipeline for training, optimizing, and evaluating a KNN classifier.
- Hyperparameter optimization visualized through:
  - Accuracy vs. `n_neighbors` plot.
  - 3D scatter plots for `n_neighbors` and `random_state`.
- Outputs a detailed classification report for performance evaluation.

## Future Improvements
- Explore other machine learning models (e.g., Random Forest, Gradient Boosting).
- Incorporate cross-validation for robust performance metrics.
- Add a feature selection step to simplify the model.
