# Customer Churn Prediction
This project focuses on predicting customer churn for a telecommunications company. Customer churn is a critical problem for businesses as it directly impacts revenue and growth.
By predicting which customers are likely to churn, companies can take proactive measures to retain them.

## Project Overview
This project implements a machine learning pipeline to predict customer churn. The process involves data loading, exploratory data analysis (EDA), data preprocessing,
model training, and evaluation. The goal is to build a robust model that can identify potential churners, thereby allowing the company to implement targeted retention strategies.

## Features

- **Data Loading**: Efficiently loads customer data from a CSV file into a pandas DataFrame.
- **Data Cleaning**: Identifies and handles missing values, specifically converting ' ' in `TotalCharges` to '0.0' and casting to float. Converts target variable 'Churn' to
   numerical representation (0 and 1).
- **Exploratory Data Analysis (EDA)**: 
    - Visualizes the distribution of numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) using histograms and box plots.
    - Analyzes the correlation between numerical features using a heatmap.
    - Visualizes the distribution of categorical features using count plots.
- **Data Preprocessing**: 
    - Applies Label Encoding to all categorical features to convert them into numerical format suitable for machine learning models.
    - Splits the data into training and testing sets.
    - Addresses class imbalance in the target variable using the Synthetic Minority Oversampling Technique (SMOTE) on the training data.
- **Model Training**: 
    - Trains and evaluates three classification models: Decision Tree, Random Forest, and XGBoost.
    - Utilizes 5-fold cross-validation to assess the performance of each model with default hyperparameters.
- **Model Evaluation**: 
    - Evaluates the best performing model (Random Forest) on the unseen test set.
    - Reports key performance metrics including Accuracy Score, Confusion Matrix, and Classification Report (Precision, Recall, F1-score).
- **Predictive System**: Demonstrates how to load the saved model and encoders to make predictions on new, unseen customer data.

## Model Performance

After training various models and utilizing SMOTE to handle class imbalance, the **Random Forest Classifier** demonstrated the most promising performance with default parameters. The evaluation metrics on the test set are as follows:

- **Accuracy Score**: 0.779
- **Confusion Matrix**:
    ```
    [[875 161]
     [150 223]]
    ```
    - True Negatives (Correctly predicted No Churn): 875
    - False Positives (Incorrectly predicted Churn): 161
    - False Negatives (Incorrectly predicted No Churn): 150
    - True Positives (Correctly predicted Churn): 223
- **Classification Report**:
    ```
                  precision    recall  f1-score   support

               0       0.85      0.84      0.85      1036
               1       0.58      0.60      0.59       373

        accuracy                           0.78      1409
       macro avg       0.72      0.72      0.72      1409
    weighted avg       0.78      0.78      0.78      1409
    ```

The model shows a good balance between precision and recall for both classes, indicating its capability to identify churners effectively while maintaining a reasonable overall accuracy.
