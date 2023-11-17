import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import streamlit as stream
from tensorflow import keras 
from create_model import create_model
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'


# Load the pre-trained model
with open("optimized_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
    
# Load the scaler model
with open("scaler.pkl", "rb") as model_file:
    scaler = pickle.load(model_file)

# Load the label model
with open("label.pkl", "rb") as model_file:
    label = pickle.load(model_file)

# Streamlit app
st.title("Telecom Churning Prediction")

# User input 'Contract', 'tenure', 'OnlineSecurity', 'TotalCharges', 'MonthlyCharges', 'TechSupport'
contract_options = ["Month-to-month", "One year", "Two year"]
Contract = st.selectbox("Contract", contract_options)
contract_mapping = {"Month-to-month": 0, "One year": 1, "Two year": 2}
Contract = contract_mapping[Contract]

tenure = st.number_input("Tenure", min_value=0)
online_security_options = ["No", "Yes"]
OnlineSecurity = st.selectbox("Online Security", online_security_options)
online_security_mapping = {"No": 0, "Yes": 1}
OnlineSecurity = online_security_mapping[OnlineSecurity]

TotalCharges = st.number_input(
    "TotalCharges", min_value=0, value=10000
)
MonthlyCharges = st.slider("MonthlyCharges", min_value=0, max_value=500, value=50)
tech_support_options = ["No", "Yes"]
TechSupport = st.selectbox("Tech Support", tech_support_options)
tech_support_mapping = {"No": 0, "Yes": 1}
TechSupport = tech_support_mapping[TechSupport]


# Numpy Array
data = np.array(
    [[  Contract,
        tenure,
        OnlineSecurity,
        TotalCharges,
        MonthlyCharges,
        TechSupport,
    ]]
)
# Data Frame
df = pd.DataFrame(
    data,
    columns=[
        " Contract",
        "tenure",
        "OnlineSecurity",
        "TotalCharges",
        "MonthlyCharges",
        "TechSupport",
    
    ]
)
df['Contract'] = label_encoder_contract.fit_transform(df['Contract'])
df['OnlineSecurity'] = label_encoder_online_security.fit_transform(df['OnlineSecurity'])
df['TechSupport'] = label_encoder_tech_support.fit_transform(df['TechSupport'])
# Scaling the data
scaled_df = scaler.transform(df)
# Make predictions
prediction = model.predict(scaled_df)[0]

# Display prediction
st.subheader("Churning Prediction:")
st.write(f"The predicted churn result is: {prediction}")

st.write(
    "This is a simple Streamlit web app to demonstrate the churning features for a telecoms company."
)
