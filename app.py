import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "gbrt_pipeline.pkl")

REQUIRED_COLUMNS = [
    "Brand", "Model", "Body", "Color", "Fuel",
    "Kilometers", "Engine", "Transmission", "Gov", "Year",
]

# Load the full pipeline (preprocessor + regressor) once at startup
pipeline = joblib.load(MODEL_PATH)

app = Flask(__name__)


@app.route("/")
def home():
    return "API is running successfully."


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    # Accept a single object or a list of objects
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not payload:
        return jsonify({"error": "Send a JSON object or a non-empty list of objects"}), 400

    try:
        data = pd.DataFrame(payload)
    except Exception as e:
        return jsonify({"error": f"Invalid data: {e}"}), 400

    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        return jsonify({"error": f"Missing columns: {missing}"}), 400

    try:
        prediction = pipeline.predict(data[REQUIRED_COLUMNS])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    return jsonify({"Prediction": prediction.tolist()})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
