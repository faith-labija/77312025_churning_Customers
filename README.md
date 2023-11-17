# Telecom Churning Prediction App
# Overview
This is a simple Streamlit web application for predicting customer churn in a telecoms company. The app uses a pre-trained machine learning model to make predictions based on user-inputted data.
# Perequisites
Make sure you have the necessary dependencies installed 
Usage
The app requires the following pre-trained models:

optimized_model.pkl: Machine learning model for making predictions.

scaler.pkl: Scaler model for scaling input data.

label.pkl: Label encoder for encoding categorical variables.

Ensure that these model files are present in the same directory as your Streamlit app.

Run the Streamlit app using the provided instructions in the "Installation" section.

Input values for 'Contract', 'Tenure', 'Online Security', 'Total Charges', 'Monthly Charges', and 'Tech Support'.

The app will display the predicted churn result based on the input data.
# Installation
Run using streamlit run your_app_name.py
# Model Details
The machine learning model is loaded from optimized_model.pkl.
The scaler model is loaded from scaler.pkl.
The label encoder for categorical variables is loaded from label.pkl.
Input Features
Contract: Type of customer contract (Month-to-month, One year, Two years).
Tenure: Number of months the customer has stayed with the telecom company.
Online Security: Whether the customer has online security (Yes, No).
Total Charges: Total charges incurred by the customer.
Monthly Charges: Monthly charges incurred by the customer.
Tech Support: Whether the customer has tech support (Yes, No).

# Output
The app displays the predicted churn result based on the provided input.
Additional Information
This app is a demonstration of churning prediction features for a telecoms company.
