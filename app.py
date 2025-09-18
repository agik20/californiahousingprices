from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# === Load Model & Scaler dari folder model/ ===
model_path = os.path.join("model", "house_price_model.pkl")
scaler_path = os.path.join("model", "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array([[
        data["longitude"],
        data["latitude"],
        data["housing_median_age"],
        data["total_rooms"],
        data["total_bedrooms"],
        data["population"],
        data["households"],
        data["median_income"]
    ]])

    # Scaling
    features_scaled = scaler.transform(features)

    # Prediksi
    prediction = model.predict(features_scaled)[0]
    return jsonify({"predicted_price": round(prediction, 2)})

if __name__ == "__main__":
    app.run(debug=True)

