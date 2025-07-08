import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

def train_and_evaluate(input_path, models_dir):
    # Load preprocessed data
    df = pd.read_csv(input_path)
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_resampled
    )

    # Define models
    models = {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "random_forest": RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42),
        "xgboost": XGBClassifier(scale_pos_weight=1, use_label_encoder=False, eval_metric="logloss", random_state=42)
    }

    os.makedirs(models_dir, exist_ok=True)

    # Train & evaluate
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"\n--- Training {name} ---")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Evaluate
            auc_score = roc_auc_score(y_test, y_prob)
            print(classification_report(y_test, y_pred))
            print(f"AUC: {auc_score:.4f}")

            # Log metrics & params
            mlflow.log_param("model_name", name)
            mlflow.log_metric("auc", auc_score)

            # Log model
            mlflow.sklearn.log_model(model, f"{name}_model")

            # Save locally as well
            model_path = os.path.join(models_dir, f"{name}.joblib")
            joblib.dump(model, model_path)
            print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_evaluate(
        input_path="data/processed/undersampled_data.csv",
        models_dir="src/models/saved_models/"
    )
