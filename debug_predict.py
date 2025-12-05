import joblib
import pandas as pd
import numpy as np
import os

# Load model assets
model_path = os.path.join("models", "logistic_regression_model.joblib")
scaler_path = os.path.join("models", "standard_scaler_lr.joblib")
encoders_path = os.path.join("models", "label_encoders_lr.joblib")

print("Loading model assets...")
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoders = joblib.load(encoders_path)

print("Model loaded successfully:", type(model))
print("Scaler loaded successfully:", type(scaler))
print("Encoders loaded successfully:", type(label_encoders))

# Test data
test_data = {
    "loan_limit": "cf",
    "Gender": "Male",
    "approv_in_adv": "nopre",
    "loan_type": "type1",
    "loan_purpose": "p1",
    "Credit_Worthiness": "l1",
    "open_credit": "nopc",
    "business_or_purpose": "b/c",
    "Neg_ammortization": "not_neg",
    "interest_only": "not_int",
    "lump_sum_payment": "not_lpsm",
    "Occupancy_Type": "pr",
    "Secured_by": "home",
    "total_units": "1U",
    "credit_type": "EXP",
    "co_applicant_credit_type": "CIB",
    "age": "25-34",
    "submission_of_application": "to_inst",
    "Region": "south",
    "Security_Type": "direct",
    "rate_of_interest": 4.5,
    "Interest_rate_spread": 0.5,
    "Upfront_charges": 1500.0,
    "term": 360.0,
    "income": 7000.0,
    "Credit_Score": 720.0,
    "LTV": 75.0,
    "dtir1": 38.0
}

print("\nCreating DataFrame...")
df = pd.DataFrame([test_data])
print("DataFrame created successfully")

# Check categorical and numerical columns
EXPECTED_FEATURES = ['loan_limit', 'Gender', 'approv_in_adv', 'loan_type', 'loan_purpose', 'Credit_Worthiness', 'open_credit', 'business_or_purpose', 'Neg_ammortization', 'interest_only', 'lump_sum_payment', 'Occupancy_Type', 'Secured_by', 'total_units', 'credit_type', 'co_applicant_credit_type', 'age', 'submission_of_application', 'Region', 'Security_Type', 'rate_of_interest', 'Interest_rate_spread', 'Upfront_charges', 'term', 'income', 'Credit_Score', 'LTV', 'dtir1']
NUMERICAL_COLS = ['rate_of_interest', 'Interest_rate_spread', 'Upfront_charges', 'term', 'income', 'Credit_Score', 'LTV', 'dtir1']
CATEGORICAL_COLS = list(set(EXPECTED_FEATURES) - set(NUMERICAL_COLS))

print("\nProcessing categorical columns...")
for col in CATEGORICAL_COLS:
    le = label_encoders.get(col)
    if le and col in df.columns:
        print(f"Processing column: {col}")
        def transform_with_handling(val):
            return le.transform([val])[0] if val in le.classes_ else -1
        df[col] = df[col].apply(transform_with_handling)

print("Categorical columns processed")

print("\nProcessing numerical columns...")
if NUMERICAL_COLS:
    df[NUMERICAL_COLS] = scaler.transform(df[NUMERICAL_COLS])

print("Numerical columns processed")

print("\nCreating final processed DataFrame...")
processed = pd.DataFrame(df[EXPECTED_FEATURES])
print("Final DataFrame shape:", processed.shape)

print("\nMaking prediction...")
try:
    prediction = int(model.predict(processed)[0])
    probability = float(model.predict_proba(processed)[0][1])
    print("Prediction:", prediction)
    print("Probability:", probability)
    print("SUCCESS: Prediction completed without errors")
except Exception as e:
    print("ERROR in prediction:", str(e))
    import traceback
    traceback.print_exc()