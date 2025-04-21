import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Read Dataset
@st.cache
def load_data():
    # Ensure your `insurance.csv` file is in the correct directory
    # Adjust the path if necessary
    return pd.read_csv('insurance_data.csv')

# Load the dataset
insurance_data = load_data()

# Dashboard selection
selected_dashboard = st.sidebar.selectbox(
    "Select Dashboard",
    ["🏠 Home", "🏥 Insurance Data"]
)

if selected_dashboard == "🏥 Insurance Data":
    st.title(":bar_chart: Insurance Data Analytics Dashboard")
    st.markdown("### Analyze customer health data and claim amounts for better insurance management.")
    
    # Dataset Overview
    st.markdown("### Dataset Overview")
    st.dataframe(insurance_data.head())
    
    # Summary Statistics
    st.markdown("### Summary Statistics of the Data")
    st.write(insurance_data.describe())
    
    # Individual Analytics
    st.subheader(":mag: Search Individual Analytics")
    name = st.text_input("Enter the Name:")
    if name:
        person_data = insurance_data[insurance_data['Name'].str.contains(name, case=False)]
        if not person_data.empty:
            st.write(person_data)
            # Predict based on Premium Amount
            if 'Premium_Amount' in person_data.columns and 'Claim_Amount' in insurance_data.columns:
                premium = person_data['Premium_Amount'].values[0]
                # Simple model logic (placeholder)
                claim_prediction_model = LinearRegression()
                X = insurance_data[['Premium_Amount']]
                y = insurance_data['Claim_Amount']
                claim_prediction_model.fit(X, y)
                predicted_claim = claim_prediction_model.predict([[premium]])[0]
                st.success(f"Predicted Claim Amount: ${predicted_claim:.2f}")
        else:
            st.warning("No data found for the given name.")
    
    # Defaulters
    st.subheader(":warning: Defaulters")
    if st.button("Show Defaulters"):
        if 'Is_Defaulter' in insurance_data.columns:
            defaulters = insurance_data[insurance_data['Is_Defaulter'] == 1]
            st.write(defaulters[['Name', 'Premium_Amount', 'Claim_Amount']])
        else:
            st.error("Defaulters data is not available in the dataset.")
    
    # Age Distribution
    st.subheader(":bust_in_silhouette: Age Distribution")
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.histplot(insurance_data['Age'], bins=20, kde=True, ax=ax, color="darkgreen")
    ax.set_title("Age Distribution of Insurance Customers", fontsize=12)
    ax.set_xlabel("Age", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    st.pyplot(fig)
    
    # Health Risk vs Claim Amount
    st.subheader(":bar_chart: Health Risk vs. Claim Amount Analysis")
    if 'Health_Risk_Score' in insurance_data.columns and 'Claim_Amount' in insurance_data.columns:
        fig = px.scatter(
            insurance_data, 
            x='Health_Risk_Score', 
            y='Claim_Amount', 
            color='Region' if 'Region' in insurance_data.columns else None, 
            size='Age' if 'Age' in insurance_data.columns else None, 
            title="Health Risk vs. Claim Amount by Region",
            labels={"Health_Risk_Score": "Health Risk Score", "Claim_Amount": "Claim Amount"},
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Health Risk Score or Claim Amount data is missing in the dataset.")
    
    # Predictive Insights
    st.subheader(":crystal_ball: Predictive Insights (Regression Analysis)")
    if 'Health_Risk_Score' in insurance_data.columns and 'Claim_Amount' in insurance_data.columns:
        X = insurance_data[['Health_Risk_Score']]
        y = insurance_data['Claim_Amount']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.scatter(X_test, y_test, label="Actual Values", alpha=0.6)
        ax.plot(X_test, y_pred, color='red', label="Predicted Values", linewidth=2)
        ax.set_title("Health Risk Score vs. Predicted Claim Amount", fontsize=12)
        ax.set_xlabel("Health Risk Score", fontsize=10)
        ax.set_ylabel("Claim Amount", fontsize=10)
        ax.legend(fontsize=10)
        st.pyplot(fig)
    else:
        st.error("Required columns for regression analysis are missing in the dataset.")
