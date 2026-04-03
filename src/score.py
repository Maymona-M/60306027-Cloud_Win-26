import json
import os
import joblib
import numpy as np

model = None

def init():
    global model
    model_dir = os.environ.get("AZUREML_MODEL_DIR", ".")
    # Find model.pkl anywhere under model dir
    for root, dirs, files in os.walk(model_dir):
        for f in files:
            if f == "model.pkl":
                model_path = os.path.join(root, f)
                model = joblib.load(model_path)
                print(f"Model loaded from {model_path}")
                return
    raise FileNotFoundError(f"model.pkl not found under {model_dir}")

def run(raw_data):
    try:
        data = json.loads(raw_data)
        X = np.array(data["data"], dtype=np.float32)
        preds = model.predict(X)
        return {"predictions": preds.tolist()}
    except Exception as e:
        return {"error": str(e)}
