# ================================
# TRAIN MODEL (FINAL)
# ================================

import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Ensure model folder exists
os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv("data/car data.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Drop unnecessary columns
df.drop(['car_ID', 'CarName'], axis=1, inplace=True)

# Select important features
X = df[['horsepower', 'enginesize', 'citympg', 'highwaympg']]
y = df['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("model/car_price_model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("✅ Model trained & saved successfully!")