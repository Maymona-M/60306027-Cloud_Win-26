import json
import os
import joblib
import numpy as np
import pandas as pd

model = None

def init():
    global model
    model_path = os.path.join(os.environ.get("AZUREML_MODEL_DIR", "."), "model.pkl")
    model = joblib.load(model_path)
    print("Model loaded from", model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        X = np.array(data["data"], dtype=np.float32)
        preds = model.predict(X)
        return {"predictions": preds.tolist()}
    except Exception as e:
        return {"error": str(e)}
