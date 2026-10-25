import json
import os

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "json_data.txt"), encoding="utf-8") as f:
    data = json.load(f)

r = requests.post("http://127.0.0.1:5000/predict", json=data)
print(r.status_code, r.json())
