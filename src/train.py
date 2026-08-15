from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from sklearn.ensemble import RandomForestClassifier
import joblib

X, y = make_classification(
    n_samples=10000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    weights=[0.98, 0.02],
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))
print("Fraud cases in training:", y_train.sum())
print("Fraud cases in test:", y_test.sum())

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nLogistic Regression Results:")
print(classification_report(y_test, y_pred, digits=4))

roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC:", round(roc_auc, 4))



rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

print("\nRandom Forest Results:")
print(classification_report(y_test, rf_pred, digits=4))

rf_roc_auc = roc_auc_score(y_test, rf_prob)
print("ROC-AUC:", round(rf_roc_auc, 4))

joblib.dump(rf_model, "models/fraud_model.joblib")
print("Model saved to models/fraud_model.joblib")