import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load datasets
data = pd.read_csv("cleaned_amazon_dataset-2.csv")
data['Order_Date'] = pd.to_datetime(data['Order_Date'])
data['Order_Hour'] = pd.to_datetime(data['Order_Time'], format='%H:%M:%S').dt.hour

insurance_data = pd.read_csv("insurance_data.csv")

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced Analytics Dashboards",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
    <style>
        body {
            font-family: 'Helvetica Neue', sans-serif;
            background-color: #f3f3f3;
            color: #333;
        }
        .sidebar .sidebar-content {
            background-color: #212529;
            color: #ffffff;
        }
        .stButton > button {
            background-color: #007bff;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            padding: 12px 18px;
            box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.1);
        }
        .stButton > button:hover {
            background-color: #0056b3;
        }
        .main .block-container {
            padding: 20px;
        }
        h1, h2, h3, h4 {
            font-weight: 600;
            color: #212529;
        }
        .css-1d391kg {
            background-color: #ffffff;
        }
        .sidebar .sidebar-header {
            font-size: 24px;
            color: #ffffff;
        }
        .stTextInput, .stSelectbox, .stMultiselect {
            margin-top: 10px;
            padding: 8px;
            border-radius: 6px;
        }
    </style>
""", unsafe_allow_html=True)

# Navbar for dashboards
selected_dashboard = st.sidebar.radio(
    "📊 Select Dashboard:", 
    ["📦 Order Analytics", "🏥 Insurance Data", "🏦 Banking Services", "🚗 Route Optimization"]
)

# Order Analytics Dashboard
if selected_dashboard == "📦 Order Analytics":
    # Title
    st.title(":truck: Advanced Order Analytics Dashboard")
    st.markdown("### Dive deep into the performance of orders and delivery times.")

    # Sidebar filters
    st.sidebar.header("🚚 Filters & Settings")
    date_range = st.sidebar.date_input(
        "Select Date Range:", [data['Order_Date'].min(), data['Order_Date'].max()]
    )
    vehicle_filter = st.sidebar.multiselect(
        "Select Vehicles:", data['Vehicle'].unique(), default=data['Vehicle'].unique()
    )
    area_filter = st.sidebar.multiselect(
        "Select Areas:", data['Area'].unique(), default=data['Area'].unique()
    )
    category_filter = st.sidebar.multiselect(
        "Select Categories:", data['Category'].unique(), default=data['Category'].unique()
    )

    filtered_data = data[(data['Order_Date'] >= pd.Timestamp(date_range[0])) & 
                         (data['Order_Date'] <= pd.Timestamp(date_range[1])) & 
                         (data['Vehicle'].isin(vehicle_filter)) & 
                         (data['Area'].isin(area_filter)) & 
                         (data['Category'].isin(category_filter))]

    # Delivery Time Analysis
    st.subheader(":stopwatch: Delivery Time Analysis")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))  # Further reduced size
        sns.histplot(filtered_data['Delivery_Time'], bins=30, kde=True, ax=ax, color="royalblue")
        ax.set_title("Distribution of Delivery Times", fontsize=12)
        ax.set_xlabel("Delivery Time (minutes)", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        st.pyplot(fig)

    with col2:
        correlation_data = filtered_data[['Delivery_Time', 'Agent_Age', 'Agent_Rating', 'Order_Hour']]
        corr_matrix = correlation_data.corr()
        fig, ax = plt.subplots(figsize=(5, 3))  # Further reduced size
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax, fmt='.2f')
        ax.set_title("Correlation Matrix", fontsize=12)
        st.pyplot(fig)

    # Order Trends
    st.subheader(":chart_with_upwards_trend: Order Trends Over Time")
    orders_by_date = filtered_data.groupby('Order_Date').size().reset_index(name='Order_Count')
    fig = px.line(orders_by_date, x='Order_Date', y='Order_Count', title="Orders Over Time",
                  labels={"Order_Date": "Date", "Order_Count": "Number of Orders"},
                  template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Peak Order Times
    st.subheader(":clock3: Peak Order Times (Hourly Breakdown)")
    orders_by_hour = filtered_data.groupby('Order_Hour').size().reset_index(name='Order_Count')
    fig = px.bar(orders_by_hour, x='Order_Hour', y='Order_Count', title="Orders by Hour of the Day",
                 labels={"Order_Hour": "Hour of the Day", "Order_Count": "Number of Orders"},
                 template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Factors Affecting Delivery Time
    st.subheader(":thinking_face: Factors Affecting Delivery Time")
    selected_factor = st.selectbox("Select Factor to Analyze:", ["Weather", "Traffic", "Area", "Vehicle"])
    factor_impact = filtered_data.groupby(selected_factor)['Delivery_Time'].mean().reset_index()
    fig = px.bar(factor_impact, x=selected_factor, y='Delivery_Time',
                 title=f"Impact of {selected_factor} on Delivery Time", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Delivery Time Prediction
    st.subheader(":crystal_ball: Predictive Analytics")
    st.markdown("### Predict delivery times based on agent performance.")
    X = filtered_data[['Agent_Rating']]
    y = filtered_data['Delivery_Time']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(5, 3))  # Further reduced size
    ax.scatter(X_test, y_test, label="Actual Values", alpha=0.6)
    ax.plot(X_test, y_pred, color='red', label="Predicted Values", linewidth=2)
    ax.set_title("Agent Rating vs. Predicted Delivery Time", fontsize=12)
    ax.set_xlabel("Agent Rating", fontsize=10)
    ax.set_ylabel("Delivery Time (minutes)", fontsize=10)
    ax.legend(fontsize=10)
    st.pyplot(fig)

# Insurance Data Dashboard
elif selected_dashboard == "🏥 Insurance Data":
    st.title(":bar_chart: Insurance Data Analytics Dashboard")
    st.markdown("### Analyze customer health data and claim amounts for better insurance management.")

    # Display dataset overview
    st.markdown("### Dataset Overview")
    st.dataframe(insurance_data.head())

    # Summary statistics
    st.markdown("### Summary Statistics of the Data")
    st.write(insurance_data.describe())

    # Age distribution (Further reduced size)
    st.subheader(":bust_in_silhouette: Age Distribution")
    fig, ax = plt.subplots(figsize=(5, 3))  # Further reduced size
    sns.histplot(insurance_data['Age'], bins=20, kde=True, ax=ax, color="darkgreen")
    ax.set_title("Age Distribution of Insurance Customers", fontsize=12)
    ax.set_xlabel("Age", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    st.pyplot(fig)

    # Health Risk vs. Claim Amount
    st.subheader(":bar_chart: Health Risk vs. Claim Amount Analysis")
    fig = px.scatter(
        insurance_data, 
        x='Health_Risk_Score', 
        y='Claim_Amount', 
        color='Region', 
        size='Age', 
        title="Health Risk vs. Claim Amount by Region",
        labels={"Health_Risk_Score": "Health Risk Score", "Claim_Amount": "Claim Amount"},
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Predictive Insights: BMI vs Charges (Further reduced size)
    st.subheader(":crystal_ball: Predictive Insights (Regression Analysis)")
    st.markdown("### Predict Claim Amount based on Health Risk Score.")
    X = insurance_data[['Health_Risk_Score']]
    y = insurance_data['Claim_Amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(5, 3))  # Further reduced size
    ax.scatter(X_test, y_test, label="Actual Values", alpha=0.6)
    ax.plot(X_test, y_pred, color='red', label="Predicted Values", linewidth=2)
    ax.set_title("Health Risk Score vs. Predicted Claim Amount", fontsize=12)
    ax.set_xlabel("Health Risk Score", fontsize=10)
    ax.set_ylabel("Claim Amount", fontsize=10)
    ax.legend(fontsize=10)
    st.pyplot(fig)

# Banking Services Page
elif selected_dashboard == "🏦 Banking Services":
    st.title(":bank: Banking Services")
    st.markdown("### This page will contain banking analytics and tools.")

# Route Optimization Page
elif selected_dashboard == "🚗 Route Optimization":
    st.title(":world_map: Route Optimization")
    st.markdown("### This page will contain route optimization tools and analytics.")
