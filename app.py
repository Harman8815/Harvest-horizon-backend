import os
from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model from the .pkl file
try:
    with open("crop_prediction.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)

# Home route
@app.route("/")
def home():
    print("Home endpoint accessed.")
    return "Hello! Welcome to the Crop Prediction API."

@app.route("/predict", methods=["POST"])
def predict():
    # Step 1: Extract data from the request
    try:
        data = request.json
        print("Request data received:", data)
    except Exception as e:
        print("Error extracting JSON data:", e)
        return jsonify({"error": "Invalid JSON data"}), 400

    # Step 2: Convert data to DataFrame
    try:
        input_data = pd.DataFrame([data])  # Convert to DataFrame for model input
        print("Data converted to DataFrame:", input_data)
    except Exception as e:
        print("Error converting data to DataFrame:", e)
        return jsonify({"error": "Data conversion failed"}), 500

    # Step 3: Make a prediction
    try:
        prediction = model.predict(input_data)[0]
        print("Prediction result:", prediction)
        return jsonify({"prediction": prediction})
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting app on port {port}")
    app.run(host="0.0.0.0", port=port)
