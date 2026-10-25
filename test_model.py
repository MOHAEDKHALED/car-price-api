import os
import sys

import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "gbrt_pipeline.pkl")

try:
    pipeline = joblib.load(MODEL_PATH)
except Exception as err:
    print(f"Failed to load model: {err!r}")
    print("Check that you are using Python 3.11 and scikit-learn==1.2.2")
    sys.exit(1)

new_data = pd.DataFrame({
    "Brand": ["Hyundai", "Fiat"],
    "Model": ["Accent", "Punto"],
    "Body": ["Sedan", "Hatchback"],
    "Color": ["Blue", "Red"],
    "Fuel": ["Gasoline", "Diesel"],
    "Kilometers": ["40000 to 159999", "80000 to 199999"],
    "Engine": ["1.6L", "1.3L"],
    "Transmission": ["Automatic", "Manual"],
    "Gov": ["Yes", "No"],
    "Year": [2018, 2017],
})

predictions = pipeline.predict(new_data)
print(f"Predicted Prices: {predictions}")
