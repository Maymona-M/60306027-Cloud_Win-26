from fastapi import FastAPI
from src.predict.predict import Passenger, predict_survival

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predict survival using a scikit-learn model saved with joblib.",
    version="1.0.0",
)

@app.get("/")
def read_root():
    return {
        "message": "Titanic Survival Prediction API running. POST /predict with passenger JSON.",
        "artifact_support": [".joblib"]
    }

@app.post("/predict")
def predict_survival_endpoint(passenger: Passenger):
    """
    Accepts passenger data in JSON format and returns predicted survival.
    Example JSON:
    {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }
    """
    prediction = predict_survival(passenger)
    return {"survived": prediction}
