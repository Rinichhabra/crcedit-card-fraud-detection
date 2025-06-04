import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Credit Card Fraud Detection")

# Features the model was trained on (based on your earlier test)
features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9']

# Input form
input_data = []
for feature in features:
    val = st.number_input(f'Enter value for {feature}', format="%.6f")
    input_data.append(val)

input_array = np.array(input_data).reshape(1, -1)

# Predict button
if st.button('Predict Fraud'):
    prediction = model.predict(input_array)[0]
    proba = model.predict_proba(input_array)[0][1]
    
    if prediction == 1:
        st.error(f"Fraudulent Transaction Detected! 🚨 Probability: {proba:.2%}")
    else:
        st.success(f"Transaction is Legitimate ✅ Probability of fraud: {proba:.2%}")


st.sidebar.info("""
Developed by **Rini Chhabra**  
Email: rinisamuel27@gmail.com  
[GitHub](https://github.com/Rinichhabra) | [LinkedIn](https://linkedin.com/in/rini-chhabra-06606a221)
""")
