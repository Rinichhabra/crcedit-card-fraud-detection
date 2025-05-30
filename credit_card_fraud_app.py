import streamlit as st
import pickle
import numpy as np

# Load the trained XGBoost model
model = pickle.load(open('xgb_model.pkl', 'rb'))

# App title and description
st.title("💳 Credit Card Fraud Detection")
st.write("""
This app predicts whether a given credit card transaction is fraudulent based on selected input features.
Please enter the feature values below and click **Predict**.
""")

# User input fields
v_features = []
for i in range(1, 29):
    value = st.number_input(f"V{i}", step=0.1, format="%.4f")
    v_features.append(value)

amount = st.number_input("Transaction Amount", min_value=0.0, step=0.1, format="%.2f")

# Combine all inputs
input_data = np.array([v_features + [amount]])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"🚨 Alert: This transaction is likely **fraudulent**. Probability: {probability:.2f}")
    else:
        st.success(f"✅ Safe: This transaction is **not fraudulent**. Probability: {probability:.2f}")

# Sidebar info
st.sidebar.info("""
Developed by **Rini Chhabra**  
Email: rinisamuel27@gmail.com  
[GitHub](https://github.com/Rinichhabra) | [LinkedIn](https://linkedin.com/in/rini-chhabra-06606a221)
""")
