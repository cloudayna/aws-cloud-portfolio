import sqlite3


DB_PATH = "data/predictions.db"


def init_db():
    connection = sqlite3.connect(DB_PATH)

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction TEXT NOT NULL,
            fraud_probability REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    connection.commit()
    connection.close()


def save_prediction(prediction, fraud_probability):
    connection = sqlite3.connect(DB_PATH)

    connection.execute(
        """
        INSERT INTO predictions (
            prediction,
            fraud_probability
        )
        VALUES (?, ?)
        """,
        (prediction, fraud_probability),
    )

    connection.commit()
    connection.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized.")