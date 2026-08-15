import joblib
import numpy as np


model = joblib.load("models/fraud_model.joblib")


def predict_transaction(features):
    transaction = np.array([features])

    prediction = model.predict(transaction)[0]
    probability = model.predict_proba(transaction)[0][1]

    if prediction == 1:
        label = "FRAUD"
    else:
        label = "NORMAL"

    return {
        "prediction": label,
        "fraud_probability": round(float(probability), 4),
    }

if __name__ == "__main__":
    sample_transaction = [
        0.5, -1.2, 0.3, 1.1, -0.7,
        0.2, 1.5, -0.4, 0.8, -1.0
    ]

    result = predict_transaction(sample_transaction)
    print(result)