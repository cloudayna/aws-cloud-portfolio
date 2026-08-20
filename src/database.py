import uuid
from datetime import datetime, timezone
from decimal import Decimal

import boto3


TABLE_NAME = "fraud-detection-predictions"
REGION = "us-east-1"

dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(TABLE_NAME)


def init_db():
    return None


def save_prediction(prediction, fraud_probability):
    prediction_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    table.put_item(
        Item={
            "prediction_id": prediction_id,
            "prediction": prediction,
            "fraud_probability": Decimal(str(fraud_probability)),
            "created_at": created_at,
        }
    )


def get_predictions():
    response = table.scan()
    items = response.get("Items", [])

    while "LastEvaluatedKey" in response:
        response = table.scan(
            ExclusiveStartKey=response["LastEvaluatedKey"]
        )
        items.extend(response.get("Items", []))

    items.sort(
        key=lambda item: item.get("created_at", ""),
        reverse=True,
    )

    for item in items:
        if "fraud_probability" in item:
            item["fraud_probability"] = float(
                item["fraud_probability"]
            )

    return items


if __name__ == "__main__":
    print("DynamoDB configured.")
