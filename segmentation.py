import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model and scaler
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title and instructions
st.title("-: Customer Segmentation :-")
st.write("Enter the customer details to predict the Segment:")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Income", min_value=1000, max_value=200000, value=50000)
total_spent = st.number_input("Total_Spend", min_value=0, max_value=100000, value=10000)
num_web_purchases = st.number_input("NumWebPurchases", min_value=0, max_value=100, value=5)
num_store_purchases = st.number_input("NumStorePurchases", min_value=0, max_value=100, value=10)
num_web_visits = st.number_input("NumWebVisitsMonth", min_value=0, max_value=1000, value=50)
recency = st.number_input("Recency", min_value=0, max_value=365, value=30)

# Prepare input data
input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total_Spend": [total_spent],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visits],
    "Recency": [recency]
})

# Scale the data
input_scaled = scaler.transform(input_data)

# Predict and display result
if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]
    st.success(f"The customer belongs to Segment: {cluster}")

    # Segment descriptions
    segment_descriptions = {
        0: "High Budget, Web Customers",
        1: "High Spending",
        2: "Web Visitors",
        3: "Store Visitors",
        4: "Low Budget, Web Customers",
        5: "Low Budget, Store Customers"
    }

    # Display segment description
    description = segment_descriptions.get(cluster, "No description available for this segment.")
    st.markdown(f"**Segment Description:** {description}")
