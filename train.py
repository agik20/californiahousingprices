import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# === Load dataset California Housing ===
# Pastikan file housing.csv ada di folder data/
data_path = os.path.join("housing.csv")
data = pd.read_csv(data_path)

# === Pilih 8 fitur numerik saja ===
X = data[[
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income"
]]
y = data["median_house_value"]

# === Split dataset ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Train model (Random Forest) ===
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# === Save model & scaler ke folder model/ ===
os.makedirs("model", exist_ok=True)
joblib.dump(model, os.path.join("model", "house_price_model.pkl"))
joblib.dump(scaler, os.path.join("model", "scaler.pkl"))

print("Model dan scaler berhasil disimpan di folder 'model/'")
