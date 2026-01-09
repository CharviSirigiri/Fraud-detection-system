import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('fraud_detection_model.pkl')

st.title("Fraud Detection Prediction App")
st.markdown("Please input the transaction details below.")

st.divider()

transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH 2OUT", "DEBIT", "DEPOSIT"])
amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance of Origin Account", min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input("New Balance of Origin Account", min_value=0.0, value=9000.0) # Fixed Typo
oldbalanceDest = st.number_input("Old Balance of Destination Account", min_value=0.0, value=5000.0)
newbalanceDest = st.number_input("New Balance of Destination Account", min_value=0.0, value=6000.0)

if st.button("Predict"):
    # 1. Create the DataFrame with the EXACT names used in training
    # Note: 'newbalanceOrig' was missing from your previous dictionary due to a typo.
    input_data = pd.DataFrame({
        'type': [transaction_type],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig], # Must match the training column name exactly
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest],
        # 2. Add the Engineered Features you created in your Notebook
        'balanceDiffOrig': [oldbalanceOrg - newbalanceOrig],
        'balanceDiffDest': [newbalanceDest - oldbalanceDest]
    })

    # Ensure the column order matches your X_train (type, amount, oldbalanceOrg, etc.)
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("The transaction is predicted to be FRAUDULENT.")
    else:
        st.success("The transaction is predicted to be LEGITIMATE.")