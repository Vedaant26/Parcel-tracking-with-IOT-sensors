import streamlit as st
import pandas as pd
import random
import time

# Simulated Data Retrieval Functions
def get_rfid_data():
    """Simulate fetching RFID-based package tracking data."""
    return {
        "RFID Tag": f"TAG-{random.randint(1000, 9999)}",
        "Location": random.choice(["Warehouse A", "In Transit", "Delivery Hub", "Delivered"]),
        "Timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }

def get_smoke_sensor_data():
    """Simulate fetching smoke sensor readings."""
    return random.uniform(0.1, 5.0)  # Simulated smoke level in ppm

# Streamlit App
st.title("Delivery Detection and Transportation Monitoring")

# RFID Tracking Section
st.header("RFID Package Tracking")
if st.button("Fetch RFID Data"):
    rfid_data = get_rfid_data()
    st.write(rfid_data)

# Smoke Sensor Monitoring Section
st.header("Smoke Sensor Monitoring")
smoke_level = get_smoke_sensor_data()
st.metric(label="Current Smoke Level (ppm)", value=f"{smoke_level:.2f}")

if smoke_level > 3.0:
    st.warning("⚠ High smoke level detected! Potential fire hazard.")
else:
    st.success("✅ Smoke levels are normal.")
    
