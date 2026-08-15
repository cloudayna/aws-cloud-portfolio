from fastapi import FastAPI
from pydantic import BaseModel


from src.predict import predict_transaction


app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0",
)


class Transaction(BaseModel):
    features: list[float]


@app.get("/")
def root():
    return {
        "status": "healthy",
        "service": "fraud-detection-api",
    }


@app.post("/predict")
def predict(transaction: Transaction):
    result = predict_transaction(transaction.features)
    return result