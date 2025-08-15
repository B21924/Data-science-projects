# app.py
import streamlit as st
import pickle
import numpy as np
import plotly.express as px

# Load model and scaler
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit page config
st.set_page_config(page_title="🏡 House Price Prediction", layout="wide")
st.title("🏠 California House Price Prediction")
st.write("Enter house details below to predict the price")

# Input fields
bedrooms = st.number_input("Bedrooms", 1, 10, 3, step=1)
balconies = st.number_input("Balconies", 0, 5, 1, step=1)
washrooms = st.number_input("Washrooms", 1, 10, 2, step=1)
sqft = st.number_input("House Area (sqft)", 200, 10000, 1200, step=50)
house_age = st.number_input("House Age (years)", 0, 10, 5, step=1)
locations = ["Downtown", "Suburb", "Countryside", "Uptown"]
location = st.selectbox("Location", locations)

# Predict button
if st.button("Predict Price"):
    with st.spinner("Predicting... ⏳"):
        loc_dict = {loc: idx for idx, loc in enumerate(locations)}
        loc_num = loc_dict[location]

        features = np.array([bedrooms, balconies, washrooms, sqft, house_age, loc_num]).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0] * 100000

    st.success(f"Predicted House Price: **${prediction:,.2f}**")

    # Dynamic Plot: Price vs Sqft
    sqft_range = np.arange(200, 10001, 200)
    pred_prices_sqft = [
        model.predict(scaler.transform(np.array([bedrooms, balconies, washrooms, s, house_age, loc_num]).reshape(1,-1)))[0]*100000
        for s in sqft_range
    ]
    fig1 = px.line(x=sqft_range, y=pred_prices_sqft, labels={'x':'Sqft', 'y':'Predicted Price ($)'},
                    title=f'Predicted Price vs House Area (Bedrooms={bedrooms})')
    st.plotly_chart(fig1, use_container_width=True)

    # Dynamic Plot: Price vs Bedrooms
    bedroom_range = np.arange(1, 11, 1)
    pred_prices_bed = [
        model.predict(scaler.transform(np.array([b, balconies, washrooms, sqft, house_age, loc_num]).reshape(1,-1)))[0]*100000
        for b in bedroom_range
    ]
    fig2 = px.line(x=bedroom_range, y=pred_prices_bed, labels={'x':'Bedrooms', 'y':'Predicted Price ($)'},
                    title=f'Predicted Price vs Bedrooms (Sqft={sqft})')
    st.plotly_chart(fig2, use_container_width=True)

    # Dynamic Plot: Price vs Location
    loc_prices = [
        model.predict(scaler.transform(np.array([bedrooms, balconies, washrooms, sqft, house_age, l_idx]).reshape(1,-1)))[0]*100000
        for l_idx in range(4)
    ]
    fig3 = px.bar(x=locations, y=loc_prices, labels={'x':'Location','y':'Predicted Price ($)'},
                    title="Predicted Price vs Location")
    st.plotly_chart(fig3, use_container_width=True)
