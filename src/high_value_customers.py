import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib


def generate_synthetic_customers(n_customers=10000, random_state=42):
    np.random.seed(random_state)
    customer_ids = np.arange(1, n_customers + 1)

    age = np.clip(np.random.normal(35, 10, n_customers).astype(int), 18, 80)
    gender = np.random.choice(["Male", "Female", "Other"], size=n_customers, p=[0.48, 0.48, 0.04])
    tenure_months = np.random.exponential(scale=24, size=n_customers).astype(int) + 1
    frequency = np.random.poisson(3, n_customers) + 1
    avg_order_value = np.random.normal(70, 30, n_customers).clip(10, 500)
    total_orders = np.maximum(1, (frequency * np.random.uniform(0.8, 1.2, n_customers)).astype(int))
    total_revenue = np.round(total_orders * avg_order_value * np.random.uniform(0.8, 1.5, n_customers), 2)
    recency_days = np.clip(np.random.exponential(scale=60, size=n_customers).astype(int), 0, 365)
    returns_rate = np.clip(np.random.beta(2, 18, n_customers), 0, 0.5)

    data = pd.DataFrame({
        "customer_id": customer_ids,
        "age": age,
        "gender": gender,
        "tenure_months": tenure_months,
        "total_orders": total_orders,
        "frequency": frequency,
        "avg_order_value": avg_order_value,
        "total_revenue": total_revenue,
        "recency_days": recency_days,
        "returns_rate": returns_rate,
    })

    data["monetary_per_order"] = (data["total_revenue"] / data["total_orders"]).round(2)
    data["rfm_score"] = (data["recency_days"] * -1) + data["frequency"] * 2 + data["monetary_per_order"] * 0.1

    threshold = np.percentile(data["total_revenue"], 80)
    data["high_value"] = (data["total_revenue"] >= threshold).astype(int)

    return data


def get_features_and_labels(data):
    encoded = pd.get_dummies(data[["gender"]], drop_first=True)
    X = pd.concat([
        data[["age", "tenure_months", "total_orders", "frequency", "avg_order_value", "total_revenue", "recency_days", "returns_rate", "monetary_per_order", "rfm_score"]],
        encoded,
    ], axis=1)
    y = data["high_value"]
    return X, y


def train_and_evaluate(data, output_dir="."):
    X, y = get_features_and_labels(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=500, solver="liblinear")
    lr.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)

    results = {}

    for name, model in [("LogisticRegression", lr), ("RandomForest", rf)]:
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_proba)

        results[name] = {
            "classification_report": report,
            "roc_auc": auc,
        }

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    joblib.dump(rf, os.path.join(output_dir, "high_value_customer_model.joblib"))

    features = X.columns.tolist()
    importance_df = pd.DataFrame({"feature": features, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
    importance_df.to_csv(os.path.join(output_dir, "feature_importances.csv"), index=False)

    return results


def run_pipeline(customers=10000):
    data = generate_synthetic_customers(customers)

    os.makedirs("data", exist_ok=True)
    data.to_csv("data/customers.csv", index=False)

    print(f"Generated {len(data)} synthetic customers and saved to data/customers.csv")

    results = train_and_evaluate(data, output_dir="models")

    for model_name, model_data in results.items():
        print(f"\n=== {model_name} ===")
        print(f"ROC AUC: {model_data['roc_auc']:.4f}")
        print(pd.DataFrame(model_data["classification_report"]).transpose())

    print("Saved model and scaler to models/high_value_customer_model.joblib and models/scaler.joblib")


if __name__ == "__main__":
    run_pipeline(10000)
