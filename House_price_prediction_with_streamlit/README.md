# 🏡 California House Price Prediction

An interactive **Streamlit web app** to predict house prices in California using **Python** and **Machine Learning**.  
The project demonstrates **feature engineering, model training, scaling, and interactive visualizations**.

---

## **Project Overview**

- Predict house prices based on:
  - Bedrooms, Washrooms, Balconies, Square Footage, House Age, and Location
- Dynamic visualizations with Plotly:
  - Predicted Price vs House Area
  - Predicted Price vs Number of Bedrooms
  - Predicted Price vs Location
- Scaled features for accurate predictions
- Streamlit interface for a **clean, user-friendly experience**

---

## **About the Pickle (.pkl) Files**

The project uses **two pickle files** to run the app:

1. `house_price_model.pkl` – Stores the trained **Random Forest model**.
2. `scaler.pkl` – Stores the **StandardScaler** used to scale input features.

> These files are **binary representations of Python objects**, allowing the app to quickly load the trained model and scaler without retraining every time.

---

## **Creating the Pickle Files**

To generate the `.pkl` files locally and run the app:

Ensure you have Python 3 and required libraries installed:
   to install libraries - pip install -r requirements.txt
   to generate pickle files - python model.py
   then run - streamlit run app.py


The .pkl files are not included in this repo due to size restrictions.
Run model.py locally to generate them before running the app.


