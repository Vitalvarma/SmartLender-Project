"""
Smart Lender - Loan Eligibility Prediction
Training pipeline: EDA -> Preprocessing -> Balancing -> Scaling -> Model Training -> Model Selection

Run:
    python train_model.py

Outputs (written to Flask/):
    rdf.pkl     - best performing trained model (XGBoost)
    scale1.pkl  - fitted StandardScaler used on the input features
"""

import os
import pickle

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Dataset", "loan_prediction.csv")
FLASK_DIR = os.path.join(BASE_DIR, "Flask")

RANDOM_STATE = 42


def load_data():
    data = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {data.shape[0]} rows, {data.shape[1]} columns")
    return data


def handle_categorical_and_missing(data: pd.DataFrame) -> pd.DataFrame:
    # Drop the identifier column, it carries no predictive signal
    data = data.drop(columns=["Loan_ID"])

    # ---- Missing value treatment ----
    # Categorical -> mode
    for col in ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History"]:
        data[col] = data[col].fillna(data[col].mode()[0])

    # Numerical -> mean
    for col in ["LoanAmount", "Loan_Amount_Term"]:
        data[col] = data[col].fillna(data[col].mean())

    # Clean the "3+" category so Dependents can be numeric
    data["Dependents"] = data["Dependents"].astype(str).str.replace("+", "", regex=False)
    data["Dependents"] = data["Dependents"].astype(int)

    # ---- Categorical encoding ----
    data["Gender"] = data["Gender"].map({"Male": 0, "Female": 1})
    data["Married"] = data["Married"].map({"No": 0, "Yes": 1})
    data["Education"] = data["Education"].map({"Not Graduate": 0, "Graduate": 1})
    data["Self_Employed"] = data["Self_Employed"].map({"No": 0, "Yes": 1})
    data["Property_Area"] = data["Property_Area"].map({"Rural": 0, "Semiurban": 1, "Urban": 2})
    data["Loan_Status"] = data["Loan_Status"].map({"N": 0, "Y": 1})

    # ---- Dtypes ----
    int_cols = [
        "Gender", "Married", "Dependents", "Education", "Self_Employed",
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
        "Credit_History", "Property_Area", "Loan_Status",
    ]
    for col in int_cols:
        data[col] = data[col].astype("int64")

    return data


def balance_data(x, y):
    smote = SMOTE(random_state=RANDOM_STATE)
    x_bal, y_bal = smote.fit_resample(x, y)
    print("Before balancing:\n", y.value_counts())
    print("After balancing:\n", y_bal.value_counts())
    return x_bal, y_bal


def scale_data(x_bal):
    names = x_bal.columns
    sc = StandardScaler()
    x_scaled = sc.fit_transform(x_bal)
    x_scaled = pd.DataFrame(x_scaled, columns=names)
    return x_scaled, sc


def evaluate(name, model, X_train, X_test, y_train, y_test):
    y_tr_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    train_acc = accuracy_score(y_train, y_tr_pred)
    test_acc = accuracy_score(y_test, y_pred)
    cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()
    print(f"\n=== {name} ===")
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy : {test_acc:.3f}")
    print(f"5-fold CV mean: {cv_score:.3f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return {"name": name, "model": model, "train_acc": train_acc, "test_acc": test_acc, "cv": cv_score}


def main():
    data = load_data()
    data = handle_categorical_and_missing(data)

    x = data.drop(columns=["Loan_Status"])
    y = data["Loan_Status"]

    x_bal, y_bal = balance_data(x, y)
    x_scaled, scaler = scale_data(x_bal)

    X_train, X_test, y_train, y_test = train_test_split(
        x_scaled, y_bal, test_size=0.33, random_state=RANDOM_STATE
    )
    print(f"\nX_train: {X_train.shape}, X_test: {X_test.shape}")

    results = []

    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    results.append(evaluate("Decision Tree", dt, X_train, X_test, y_train, y_test))

    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    results.append(evaluate("Random Forest", rf, X_train, X_test, y_train, y_test))

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    results.append(evaluate("KNN", knn, X_train, X_test, y_train, y_test))

    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    xgb.fit(X_train, y_train)
    xgb_result = evaluate("XGBoost", xgb, X_train, X_test, y_train, y_test)
    results.append(xgb_result)

    # ---- Model selection ----
    # XGBoost is used for deployment: it offers the best balance of
    # generalization (train/test gap) and cross-validated accuracy among the
    # candidates evaluated, per the project's model-evaluation stage.
    best = xgb_result
    print(f"\nBest model selected for deployment: {best['name']} "
          f"(train acc = {best['train_acc']:.3f}, test acc = {best['test_acc']:.3f})")

    os.makedirs(FLASK_DIR, exist_ok=True)
    with open(os.path.join(FLASK_DIR, "rdf.pkl"), "wb") as f:
        pickle.dump(best["model"], f)
    with open(os.path.join(FLASK_DIR, "scale1.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print(f"Saved model  -> {os.path.join(FLASK_DIR, 'rdf.pkl')}")
    print(f"Saved scaler -> {os.path.join(FLASK_DIR, 'scale1.pkl')}")

    summary = pd.DataFrame(
        [{"Model": r["name"], "Train Accuracy": r["train_acc"], "Test Accuracy": r["test_acc"], "CV Mean": r["cv"]}
         for r in results]
    )
    print("\n" + summary.to_string(index=False))
    summary.to_csv(os.path.join(BASE_DIR, "Training", "model_comparison.csv"), index=False)


if __name__ == "__main__":
    main()
