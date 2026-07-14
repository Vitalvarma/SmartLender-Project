"""
Smart Lender - Loan Eligibility Prediction
Flask web application

Run:
    python app.py
Then open the URL shown in the terminal (default http://127.0.0.1:5000/)
"""

import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "rdf.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scale1.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Mappings must match the encoding used during training (Training/train_model.py)
GENDER_MAP = {"Male": 0, "Female": 1}
MARRIED_MAP = {"No": 0, "Yes": 1}
EDUCATION_MAP = {"Not Graduate": 0, "Graduate": 1}
SELF_EMPLOYED_MAP = {"No": 0, "Yes": 1}
PROPERTY_AREA_MAP = {"Rural": 0, "Semiurban": 1, "Urban": 2}

FEATURE_ORDER = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Credit_History", "Property_Area",
]


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict")
def predict_form():
    return render_template("predict.html")


@app.route("/submit", methods=["POST"])
def submit():
    form = request.form

    dependents = form.get("Dependents", "0").replace("+", "")

    row = {
        "Gender": GENDER_MAP.get(form.get("Gender"), 0),
        "Married": MARRIED_MAP.get(form.get("Married"), 0),
        "Dependents": int(dependents),
        "Education": EDUCATION_MAP.get(form.get("Education"), 0),
        "Self_Employed": SELF_EMPLOYED_MAP.get(form.get("Self_Employed"), 0),
        "ApplicantIncome": float(form.get("ApplicantIncome", 0)),
        "CoapplicantIncome": float(form.get("CoapplicantIncome", 0)),
        "LoanAmount": float(form.get("LoanAmount", 0)),
        "Loan_Amount_Term": float(form.get("Loan_Amount_Term", 360)),
        "Credit_History": float(form.get("Credit_History", 1)),
        "Property_Area": PROPERTY_AREA_MAP.get(form.get("Property_Area"), 0),
    }

    features = pd.DataFrame([[row[col] for col in FEATURE_ORDER]], columns=FEATURE_ORDER)
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    try:
        probability = model.predict_proba(features_scaled)[0][int(prediction)]
    except AttributeError:
        probability = None

    result = "Approved" if prediction == 1 else "Rejected"

    return render_template(
        "submit.html",
        result=result,
        approved=(prediction == 1),
        probability=f"{probability * 100:.1f}%" if probability is not None else "N/A",
        applicant=row,
    )


if __name__ == "__main__":
    app.run(debug=True)
