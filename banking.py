import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def display_banking():
    st.title("Banking Services - Post India")
    
    try:
        data = pd.read_csv("banking_data.csv")
    except FileNotFoundError:
        st.warning("No existing data found. Add data to start analysis.")
        data = pd.DataFrame(columns=[
            "Account_ID", "Customer_Name", "Age", "Gender", "Marital_Status",
            "Account_Type", "Balance", "Interest_Rate", "Account_Term", 
            "Opening_Date", "Closing_Date", "Transaction_Status", 
            "Transaction_Amount", "Transaction_Type", "Frequency", 
            "Region", "Occupation", "Dependents", 
            "Customer_Satisfaction_Rating"
        ])
    tabs = st.tabs(["Add Banking Data", "Insights & Visualization", "Predictions & Analysis"])
    with tabs[0]:
        st.subheader("Add New Customer Data")
        account_id = st.text_input("Account ID")
        customer_name = st.text_input("Customer Name")
        age = st.number_input("Age", min_value=18, max_value=100, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
        account_type = st.selectbox("Account Type", ["Savings", "Current", "Fixed Deposit", "Recurring Deposit"])
        balance = st.number_input("Initial Balance", min_value=0.0, step=0.01)
        interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, step=0.01)
        account_term = st.number_input("Account Term (in years)", min_value=1, max_value=50, step=1)
        opening_date = st.date_input("Opening Date")
        transaction_status = st.selectbox("Transaction Status", ["Success", "Failed", "Pending"])
        transaction_amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
        transaction_type = st.selectbox("Transaction Type", ["Deposit", "Withdrawal", "Transfer"])
        frequency = st.selectbox("Transaction Frequency", ["Daily", "Weekly", "Monthly", "Yearly"])
        region = st.text_input("Region")
        occupation = st.text_input("Occupation")
        dependents = st.number_input("Number of Dependents", min_value=0, step=1)
        satisfaction_rating = st.slider("Customer Satisfaction Rating", 1, 5)
        
        if st.button("Submit Banking Details"):
            new_entry = {
                "Account_ID": account_id, "Customer_Name": customer_name, "Age": age,
                "Gender": gender, "Marital_Status": marital_status, "Account_Type": account_type,
                "Balance": balance, "Interest_Rate": interest_rate, "Account_Term": account_term,
                "Opening_Date": opening_date, "Closing_Date": None, "Transaction_Status": transaction_status,
                "Transaction_Amount": transaction_amount, "Transaction_Type": transaction_type,
                "Frequency": frequency, "Region": region, "Occupation": occupation, 
                "Dependents": dependents, "Customer_Satisfaction_Rating": satisfaction_rating
            }
            data = data.append(new_entry, ignore_index=True)
            data.to_csv("banking_data.csv", index=False)
            st.success("Banking details added successfully!")

    with tabs[1]:
        st.subheader("Data Insights & Visualization")
        
        if not data.empty:
            st.write("Dataset Overview:")
            st.dataframe(data)
            
            st.write("Customer Age Distribution")
            sns.histplot(data["Age"], kde=True, bins=20, color="blue")
            st.pyplot()
            
            st.write("Account Type Breakdown")
            account_type_counts = data["Account_Type"].value_counts()
            st.bar_chart(account_type_counts)
            
            st.write("Customer Satisfaction Rating Distribution")
            satisfaction_counts = data["Customer_Satisfaction_Rating"].value_counts().sort_index()
            fig, ax = plt.subplots()
            satisfaction_counts.plot(kind="bar", ax=ax, color="green")
            st.pyplot(fig)
        else:
            st.info("No data available for insights.")

    with tabs[2]:
        st.subheader("Predictive Analysis")
        
        if not data.empty:
            st.write("Satisfaction Rating vs Account Balance")
            fig, ax = plt.subplots()
            sns.scatterplot(data=data, x="Balance", y="Customer_Satisfaction_Rating", hue="Account_Type", ax=ax)
            st.pyplot(fig)
            st.write("Transaction Frequency by Region")
            region_frequency = data.groupby("Region")["Frequency"].value_counts().unstack(fill_value=0)
            st.dataframe(region_frequency)
            
            # Analysis of Transactions by Type
            st.write("Transaction Amounts by Type")
            transaction_analysis = data.groupby("Transaction_Type")["Transaction_Amount"].sum()
            st.bar_chart(transaction_analysis)
        else:
            st.info("No data available for predictions.")
