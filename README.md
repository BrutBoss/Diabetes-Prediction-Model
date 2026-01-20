# Diabetes Prediction Using Machine Learning

## 1. Problem Description

Diabetes is a chronic health condition that affects millions of people
globally and often develops gradually without noticeable early symptoms.
Early identification of individuals at risk is essential for timely
intervention, effective disease management, and the prevention of severe
complications such as heart disease, kidney failure, and vision loss.

This project aims to build a machine learning--based prediction system
that determines whether a patient is likely to have diabetes based on
their medical and demographic information. The solution can support
healthcare professionals in screening patients, prioritizing further
medical tests, and developing personalized treatment or prevention
strategies. Additionally, the project provides insights into the key
factors associated with diabetes through exploratory data analysis.

## 2. Dataset

The dataset contains medical and demographic data collected from
patients, along with a binary indicator of diabetes status.

**Target variable** - diabetes (0 = No diabetes, 1 = Diabetes)

**Features** - age - gender - bmi - hypertension - heart_disease -
smoking_history - HbA1c_level - blood_glucose_level

## 3. Project Structure

    diabetes-prediction/
    ├── README.md
    ├── notebook.ipynb
    ├── train.py
    ├── predict.py
    ├── model.bin
    ├── requirements.txt
    ├── Dockerfile
    └── data/
        └── diabetes.csv

## 4. Exploratory Data Analysis

The notebook includes data cleaning, feature analysis, visualization of
distributions, correlation analysis, and feature importance evaluation.

## 5. Model Training and Selection

Multiple models were trained and compared, including Logistic
Regression, Random Forest, and Gradient Boosting models. Hyperparameter
tuning was applied, and the best-performing model was selected.

## 6. How to Run the Project Locally

### Install dependencies

    pip install -r requirements.txt

### Train the model

    python train.py

### Run the API

    python predict.py

## 7. Using the Prediction API

POST /predict

Example request:

    {
      "age": 45,
      "gender": "Male",
      "bmi": 28.3,
      "hypertension": 1,
      "heart_disease": 0,
      "smoking_history": "former",
      "HbA1c_level": 6.1,
      "blood_glucose_level": 145
    }

Example response:

    {
      "diabetes_probability": 0.82,
      "diabetes_prediction": true
    }

## 8. Docker

Build the image:

    docker build -t diabetes-predictor .

Run the container:

    docker run -p 9696:9696 diabetes-predictor

## 9. Technologies Used

Python, Pandas, NumPy, Scikit-learn, Flask, Docker
