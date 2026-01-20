import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


DATA_PATH = "/workspaces/Diabetes-Prediction-Model/diabetes_prediction_dataset.csv"
MODEL_PATH = "model.bin"
RANDOM_STATE = 42


def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    y = df["diabetes"]
    X = df.drop(columns=["diabetes"])

    X = pd.get_dummies(X, drop_first=True)

    return X, y


def train_and_evaluate_models(X_train, y_train, X_val, y_val):
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        )
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        if name == "logistic_regression":
            y_pred = model.predict_proba(X_val)[:, 1]
        else:
            y_pred = model.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y_val, y_pred)
        results[name] = (model, auc)

        print(f"{name}: ROC-AUC = {auc:.4f}")

    return results


def tune_best_model(X_train, y_train):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=RANDOM_STATE))
    ])

    param_grid = {
        "model__C": [0.01, 0.1, 1, 10],
        "model__penalty": ["l2"]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring="roc_auc",
        cv=5
    )

    grid.fit(X_train, y_train)

    print("Best Logistic Regression parameters:", grid.best_params_)
    print("Best CV ROC-AUC:", grid.best_score_)

    return grid.best_estimator_


def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)

    print(f"Final model saved to {path}")


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Preprocessing data...")
    X, y = preprocess_data(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print("\nTraining and comparing multiple models...")
    results = train_and_evaluate_models(X_train, y_train, X_val, y_val)

    best_model_name = max(results, key=lambda k: results[k][1])
    best_auc = results[best_model_name][1]

    print(f"\nBest baseline model: {best_model_name} (ROC-AUC = {best_auc:.4f})")

    print("\nTuning best model...")
    final_model = tune_best_model(X_train, y_train)

    print("\nSaving final model...")
    save_model(final_model, MODEL_PATH)


if __name__ == "__main__":
    main()
