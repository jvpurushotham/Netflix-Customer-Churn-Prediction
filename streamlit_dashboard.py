# Import required libraries
import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('netflix_churn_model.pkl', 'rb'))

# Title
st.title("Netflix Churn Prediction App")

# Mappings
gender_map = {"Female": 0, "Male": 1, "Other": 2}
subscription_map = {"Basic": 0, "Standard": 2, "Premium": 1}
device_map = {"Desktop": 0, "Laptop": 1, "Mobile": 2, "TV": 3, "Tablet": 4}
region_map = {"Africa":0, "Asia":1, "Europe":2, "North America":3, "Oceania":4, "South America":5}
payment_map = {"Credit Card":0, "Crypto":1, "Debit Card":2, "Gift Card":3, "PayPal":4}
genre_map = {"Action":0, "Comedy":1, "Documentary":2, "Drama":3, "Horror":4, "Romance":5, "Sci-Fi":6}

# Create 2 columns
col1, col2 = st.columns(2)

# Column 1 inputs
with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    gender = st.selectbox("Gender", list(gender_map.keys()))
    subscription_type = st.selectbox("Subscription Type", list(subscription_map.keys()))
    watch_hours = st.number_input("Watch Hours per Week", min_value=0.0, value=10.0)
    last_login_days = st.number_input("Days Since Last Login", min_value=0.0, value=5.0)
    region = st.selectbox("Region", list(region_map.keys()))

# Column 2 inputs
with col2:
    device = st.selectbox("Device", list(device_map.keys()))
    monthly_fee = st.number_input("Monthly Fee", min_value=0.0, value=10.0)
    payment_method = st.selectbox("Payment Method", list(payment_map.keys()))
    number_of_profiles = st.number_input("Number of Profiles", min_value=1, max_value=10, value=1)
    favorite_genre = st.selectbox("Favorite Genre", list(genre_map.keys()))

# Button
st.markdown("---")
if st.button("Predict Churn"):

    try:
        input_features = [
            float(age),
            gender_map.get(gender, 0),
            subscription_map.get(subscription_type, 0),
            float(watch_hours),
            float(last_login_days),
            region_map.get(region, 0),
            device_map.get(device, 0),
            float(monthly_fee),
            payment_map.get(payment_method, 0),
            float(number_of_profiles),
            genre_map.get(favorite_genre, 0),
        ]

        input_array = np.array(input_features).reshape(1, -1)

        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0][1]

        if prediction == 1:
            st.error(f"Likely to Churn ({round(probability*100, 2)}%)")
        else:
            st.success(f"Not Likely to Churn ({round(probability*100, 2)}%)")

    except Exception as e:
        st.error(f"Error: {str(e)}")
