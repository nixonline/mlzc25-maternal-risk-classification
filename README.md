# Maternal Risk Classification

## Problem Description
This project predicts whether a pregnant patient is at **high risk** for complications based on clinical and demographic features.  
The solution helps healthcare professionals perform **quick risk assessments** to prioritize patient care.

The input features include age, BMI, blood pressure, blood sugar, heart rate, diabetes history, mental health indicators, and previous complications.  
The output is a **binary classification**: **Low Risk** or **High Risk**.

---

## Exploratory Data Analysis (EDA)
- **Dataset Overview:** 1,166 patient records with 12 features.  
- **Data Cleaning:** Removed rows with missing values (<5%) and invalid BMI/age entries.  
- **Feature Analysis:**
  - Summary statistics (min–max, mean, standard deviation) explored for numerical features.
  - Boxplots and histograms examined distributions and potential outliers.
  - Target variable (`risk_level`) distribution analyzed.
- **Feature Importance:**
  - **Random Forest:** BMI, diabetes, and heart rate ranked highly.
  - **XGBoost:** Diabetes-related indicators were most influential.

---

## Model Training
Two models were trained: **Random Forest** and **XGBoost**.

### Hyperparameter Tuning
- **XGBoost:** Tuned using `RandomizedSearchCV`.  
- **Random Forest:** Default settings used, paired with feature importance analysis.

### Performance Metrics

| Model          | Accuracy | Precision | Recall | F1 Score |
|----------------|----------|-----------|--------|----------|
| Random Forest  | 0.987    | 0.987     | 0.987  | 0.987    |
| XGBoost        | 0.996    | 0.996     | 0.996  | 0.996    |

**Observation:** XGBoost achieved slightly higher performance and is used for deployment.

---

## Exporting Notebook to Script
The training workflow is implemented in `train.py`, which:
- Loads and cleans the dataset  
- Trains the XGBoost model using the best hyperparameters  
- Saves the trained model as `xgb_model.pkl`  

The prediction workflow is implemented in `predict.py`, which serves the model via Flask.

---

## Reproducibility
- Dataset can be downloaded from Kaggle using `kagglehub`.  
- Notebook and scripts run end-to-end without errors.  
- All dependencies are listed in `requirements.txt`.

---

## Model Deployment
- A Flask app serves the trained model through a simple HTML form.  
- Live deployment (Render): **https://maternal-risk-classification.onrender.com/**

---

## Dependency & Environment Management
Use a virtual environment for consistent results:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Containerization
- Dockerfile included to build a containerized version of the app:
```bash
docker build -t maternal-risk-app .
docker run -p 5000:5000 maternal-risk-app
```
- Can be deployed locally or to any cloud service.

---

## Cloud Deployment
- Deployed on Render free tier using a Docker-based web service.
- Public endpoint available for testing (link above).
