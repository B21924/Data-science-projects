# model.py
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle


# Load California Housing dataset
data = fetch_california_housing(as_frame=True)
df = data.frame.copy()


# Feature Engineering
# Bedrooms → Round AveBedrms
df['bedrooms'] = df['AveBedrms'].round()

# Washrooms → Approximate as 1.5x AveBedrms
df['washrooms'] = (df['AveBedrms'] * 1.5).round()

# Balconies → Random small integers (0-3)
np.random.seed(42)
df['balconies'] = np.random.randint(0, 4, df.shape[0])

# Sqft → Approximate using AveRooms * AveOccup * 100
df['sqft'] = (df['AveRooms'] * df['AveOccup'] * 100).round()

# House age → Clip to max 10 years
df['house_age'] = df['HouseAge'].clip(upper=10)

# Location → Simple categorical based on latitude/longitude
def map_location(lat, lon):
    if lat > 38:
        return 0  # Countryside
    elif lat > 37.5:
        return 1  # Suburb
    elif lon < -122:
        return 2  # Downtown
    else:
        return 3  # Uptown

df['location'] = df.apply(lambda x: map_location(x['Latitude'], x['Longitude']), axis=1)


# Features and Target
X = df[['bedrooms','balconies','washrooms','sqft','house_age','location']]
y = df['MedHouseVal']


# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train Random Forest Model
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)


# Evaluate Model
y_pred = model.predict(X_test_scaled)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained successfully!")
print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")


# Save Model and Scaler
pickle.dump(model, open("house_price_model.pkl","wb"))
pickle.dump(scaler, open("scaler.pkl","wb"))
