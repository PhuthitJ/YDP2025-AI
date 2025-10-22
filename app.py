from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

# Initialize app
app = FastAPI(title="Bank Churn Prediction API")

# Define input data format
class CustomerData(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Complain: int
    Satisfaction_Score: float
    Card_Type: str
    Point_Earned: float

@app.on_event("startup")
def load_model():
    model_path = Path(__file__).with_name("bank_churn_pipeline.pkl")
    if not model_path.exists():
        raise RuntimeError(f"Model file not found at {model_path}")
    app.state.model = joblib.load(model_path)

@app.get("/")
def home():
    return {"message": "Welcome to Bank Churn Prediction API!"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": hasattr(app.state, "model")}

@app.post("/predict")
def predict_churn(data: CustomerData):
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    # Convert result to readable text
    result = "Will Churn" if int(prediction) == 1 else "Will Stay"

    return {
        "prediction": result,
        "churn_probability": round(float(probability), 3)
    }
