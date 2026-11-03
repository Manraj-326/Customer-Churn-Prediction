import streamlit as st
import pandas as pd
import joblib

# Load trained model
model_data = joblib.load("customer_churn_model.pkl")

model = model_data["model"]
threshold = model_data["threshold"]
features = model_data["features"]

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Application title
st.title("📊 Customer Churn Prediction System")

st.write(
    "Enter customer details below to predict "
    "whether the customer is likely to leave the service."
)

st.divider()

# Customer information
st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        ["Yes", "No"]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure Months",
        min_value=0,
        max_value=100,
        value=12
    )

    phone = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col2:

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )

    latitude = st.number_input(
        "Latitude",
        value=37.0
    )

    longitude = st.number_input(
        "Longitude",
        value=-100.0
    )

# Prediction button
st.divider()

if st.button("Predict Customer Churn", use_container_width=True):

    input_data = {
        "Latitude": latitude,
        "Longitude": longitude,
        "Gender": gender,
        "Senior Citizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure Months": tenure,
        "Phone Service": phone,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Contract": contract,
        "Paperless Billing": paperless,
        "Payment Method": payment,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges
    }

    # Arrange features in the original training order
    input_df = pd.DataFrame(
        [input_data],
        columns=features
    )

    # Predict probability
    probability = model.predict_proba(input_df)[0][1]

    # Apply selected threshold
    prediction = int(probability >= threshold)

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Classification Threshold",
            f"{threshold:.2f}"
        )

    if prediction == 1:

        st.error("⚠️ Customer is likely to churn!")

        st.write(
            "Recommendation: Consider contacting this customer "
            "and offering suitable retention incentives."
        )

    else:

        st.success("✅ Customer is unlikely to churn!")

        st.write(
            "The customer is predicted to continue using "
            "the service."
        )