from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict_transaction

from src.database import init_db, save_prediction, get_predictions

app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0",
)

init_db()


class Transaction(BaseModel):
    features: list[float] = Field(min_length=10, max_length=10)


@app.get("/")
def root():
    return {
        "status": "healthy",
        "service": "fraud-detection-api",
    }


@app.post("/predict")
def predict(transaction: Transaction):
    result = predict_transaction(transaction.features)

    save_prediction(
        result["prediction"],
        result["fraud_probability"],
    )

    return result

    @app.get("/predictions")
def predictions():
    return get_predictions()