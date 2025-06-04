import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("💳 Credit Card Fraud Detection")

st.write("""
This app predicts whether a credit card transaction is fraudulent based on 10 selected input features.
Adjust the sliders below and click **Predict Fraud**.
""")

# Features the model was trained on with some example ranges (you can adjust these)
features = {
    'Time': (0.0, 172792.0, 50000.0),  # min, max, default
    'V1': (-5.0, 5.0, 0.0),
    'V2': (-5.0, 5.0, 0.0),
    'V3': (-5.0, 5.0, 0.0),
    'V4': (-5.0, 5.0, 0.0),
    'V5': (-5.0, 5.0, 0.0),
    'V6': (-5.0, 5.0, 0.0),
    'V7': (-5.0, 5.0, 0.0),
    'V8': (-5.0, 5.0, 0.0),
    'V9': (-5.0, 5.0, 0.0)
}

input_data = []
for feature, (min_val, max_val, default) in features.items():
    val = st.slider(f"Enter value for {feature}", min_value=float(min_val), max_value=float(max_val), value=float(default), step=0.01)
    input_data.append(val)

input_array = np.array(input_data).reshape(1, -1)

if st.button("Predict Fraud"):
    prediction = model.predict(input_array)[0]
    proba = model.predict_proba(input_array)[0][1]
    proba_percent = proba * 100

    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! Probability: {proba_percent:.2f}%")
    else:
        st.success(f"✅ Transaction is Legitimate. Probability of fraud: {proba_percent:.2f}%")

st.sidebar.info("""
Developed by **Rini Chhabra**  
Email: rinisamuel27@gmail.com  
[GitHub](https://github.com/Rinichhabra) | [LinkedIn](https://linkedin.com/in/rini-chhabra-06606a221)
""")
