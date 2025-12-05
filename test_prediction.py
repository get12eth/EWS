import requests
import json

# Sample loan data for testing
loan_data = {
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

# Make a POST request to the predict endpoint
try:
    response = requests.post("http://localhost:8000/predict", json=loan_data)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("Prediction Result:")
        print(json.dumps(result, indent=2))
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")