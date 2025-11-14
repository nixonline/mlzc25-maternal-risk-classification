# Maternal Risk Classification
##Problem Description

This project predicts whether a pregnant patient is at high risk for complications based on clinical and demographic features.
The solution is designed to support healthcare professionals by providing quick risk assessment for prioritizing patient care.

The input features include age, BMI, blood pressure, blood sugar, heart rate, diabetes history, mental health indicators, and previous complications. The output is a binary classification: Low Risk or High Risk.

## Exploratory Data Analysis (EDA)

Dataset Overview: 1205 patient records with 12 features.

Data Cleaning: Removed rows with missing values (<5%) and invalid BMI or age entries.

Feature Analysis:
- Numerical features analyzed via min-max, mean, and standard deviation.
- Boxplots and histograms used to examine distributions and detect outliers.
- Target variable (risk_level) distribution analyzed.

Feature Importance:
- Random Forest: BMI, diabetes, and heart rate most influential.
- XGBoost: Diabetes-related features prioritized.

## Model Training

- Two models were trained: Random Forest and XGBoost.

- Hyperparameter Tuning:

    - XGBoost parameters optimized using RandomizedSearchCV.

    - Random Forest used default settings with feature importance analysis.

Performance Metrics:

Model	Accuracy	Precision	Recall	F1 Score
Random Forest	0.987	0.987	0.987	0.987
XGBoost	0.996	0.996	0.996	0.996

- Observation: XGBoost achieved slightly higher accuracy and is used for deployment.

## Exporting Notebook to Script

- Training logic is exported to train.py, which:

- Loads and cleans data

- Trains the XGBoost model with the best hyperparameters

- Saves the trained model as xgb_model.pkl

- Prediction logic is in predict.py and serves the model via Flask.

## Reproducibility

- Dataset can be downloaded via Kaggle using kagglehub.

- Notebook and scripts can be executed without errors.

- requirements.txt lists all dependencies.

## Dependency and Environment Management

- requirements.txt provided.

- irtual environment recommended for reproducibility:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

