# Netflix Customer Churn Prediction

## Overview

This project focuses on predicting customer churn for a subscription-based streaming service using machine learning techniques. By analyzing user behavior, subscription details, and engagement patterns, the model identifies customers who are likely to cancel their subscriptions.

The goal is to help businesses take proactive measures to retain customers and improve overall user satisfaction.

---

## Problem Statement

Customer churn is a major challenge for subscription-based platforms. Identifying customers who are likely to leave allows companies to:

* Reduce revenue loss
* Improve customer retention strategies
* Enhance user engagement

This project builds a predictive model to classify whether a customer will churn or not.

---

## Features of the Project

* Data preprocessing and cleaning
* Exploratory Data Analysis (EDA)
* Feature engineering
* Machine learning model building using LightGBM
* Model evaluation using multiple metrics
* Model serialization using pickle
* Deployment-ready structure with Streamlit dashboard

---

## Tech Stack

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* LightGBM
* Streamlit

---

## Project Structure

```
Netflix-Customer-Churn-Prediction/
│
├── templates/                     
├── app.py                        
├── streamlit_dashboard.py        
├── netflix-customer-churn-prediction.ipynb  
├── netflix_churn_model.pkl       
├── netflix_customer_churn.csv    
├── requirements.txt              
├── runtime.txt                   
├── dockerfile                    
├── Procfile                      
├── README.md                     
└── LICENSE                       
```

---

## Project Flow

1. Data Collection

   * Load customer dataset containing demographics, subscription, and usage data

2. Data Preprocessing

   * Handle missing values
   * Encode categorical variables using LabelEncoder
   * Feature scaling if required

3. Exploratory Data Analysis

   * Analyze churn distribution
   * Visualize relationships between features and churn

4. Feature Engineering

   * Select important features influencing churn
   * Transform variables for better model performance

5. Model Building

   * Split dataset into training and testing sets
   * Train LightGBM classifier

6. Model Evaluation

   * Accuracy Score
   * Confusion Matrix
   * Classification Report
   * ROC-AUC Score

7. Model Saving

   * Save trained model using pickle

8. Deployment

   * Build interactive UI using Streamlit
   * Serve predictions through a web interface

---

## Installation and Setup

### 1. Clone the Repository

```
git clone https://github.com/your-username/Netflix-Customer-Churn-Prediction.git
cd Netflix-Customer-Churn-Prediction
```

### 2. Create Virtual Environment

```
python -m venv venv
source venv/bin/activate     # On Mac/Linux
venv\Scripts\activate        # On Windows
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

## Running the Project

### Run Streamlit Dashboard

```
streamlit run streamlit_dashboard.py
```

### Run Flask App (if applicable)

```
python app.py
```

---

## Model Details

* Algorithm: LightGBM Classifier
* Evaluation Metrics:

  * Accuracy
  * Precision, Recall, F1-score
  * ROC-AUC

---

## Use Cases

* Subscription platforms (OTT, SaaS)
* Customer retention analytics
* Marketing strategy optimization

---

## Future Improvements

* Hyperparameter tuning for better accuracy
* Use of deep learning models
* Real-time prediction API
* Integration with cloud deployment (AWS/GCP)
* Advanced feature engineering

---

## License

This project is licensed under the MIT License.
