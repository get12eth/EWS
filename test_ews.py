#!/usr/bin/env python3
"""
Test script for the Early Warning System enhancements.
This script tests the new API endpoints and functionality.
"""

import requests
import json
from datetime import datetime

# Base URL for the API
BASE_URL = "http://localhost:8000"

# Create a session to handle cookies
session = requests.Session()

def login():
    """Login to the system to get authentication cookies."""
    # Default credentials from the app
    login_data = {
        "username": "admin",
        "password": "password"
    }
    
    try:
        response = session.post(f"{BASE_URL}/admin-login", data=login_data)
        if response.status_code == 200 or response.status_code == 302:
            print("Login successful")
            return True
        else:
            print(f"Login failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Login error: {e}")
        return False

def test_macro_indicators():
    """Test adding and retrieving macroeconomic indicators."""
    print("Testing macroeconomic indicators...")
    
    # Add macro indicators
    macro_data = {
        "gdp_growth_rate": 2.5,
        "market_index_yoy": 5.2,
        "unemployment_rate": 3.8,
        "inflation_rate": 2.1,
        "interest_rate": 3.5
    }
    
    response = session.post(f"{BASE_URL}/api/macro_indicators", json=macro_data)
    print(f"Add macro indicators response: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.json()}")
    
    # Get macro indicators
    response = session.get(f"{BASE_URL}/api/macro_indicators")
    print(f"Get macro indicators response: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.json()}")

def test_alerts():
    """Test alerts functionality."""
    print("\nTesting alerts...")
    
    # Get alerts
    response = session.get(f"{BASE_URL}/api/alerts")
    print(f"Get alerts response: {response.status_code}")
    if response.status_code == 200:
        alerts = response.json()
        print(f"Found {len(alerts)} alerts:")
        for alert in alerts:
            print(f"  - ID: {alert['id']}, Signal: {alert['risk_signal']}, Severity: {alert['severity']}, Score: {alert['prediction_score']}")
    else:
        print(f"Failed to get alerts: {response.status_code}")

def test_create_test_alert():
    """Create a test alert directly via API to verify alert functionality."""
    print("\nCreating a test alert...")
    
    # We'll create a test alert by making a direct database call
    # But first, let's make a prediction to ensure we have some data
    loan_data = {
        "loan_limit": "cf",
        "Gender": "Male",
        "approv_in_adv": "nopre",
        "loan_type": "type1",
        "loan_purpose": "p1",
        "Credit_Worthiness": "l1",
        "open_credit": "nopc",
        "business_or_commercial": "nob/c",
        "Neg_ammortization": "not_neg",
        "interest_only": "not_int",
        "lump_sum_payment": "not_lpsm",
        "construction_type": "sb",
        "occupancy_type": "pr",
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
    
    response = session.post(f"{BASE_URL}/predict", json=loan_data)
    print(f"Prediction response: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction result: {result}")
        
        # Check alerts after prediction
        print("\nChecking alerts after prediction...")
        test_alerts()

def test_alert_update():
    """Test updating an alert."""
    print("\nTesting alert update...")
    
    # First get alerts
    response = session.get(f"{BASE_URL}/api/alerts")
    if response.status_code == 200:
        alerts = response.json()
        if alerts:
            # Update the first alert
            alert_id = alerts[0]['id']
            update_data = {
                "status": "Under Review",
                "manager_notes": "Reviewing this high-risk case",
                "resolution_action": "Contact customer for additional documentation"
            }
            
            update_response = session.post(f"{BASE_URL}/api/alerts/{alert_id}/update_status", json=update_data)
            print(f"Update alert response: {update_response.status_code}")
            if update_response.status_code == 200:
                print(f"Response: {update_response.json()}")
                
                # Verify the update
                verify_response = session.get(f"{BASE_URL}/api/alerts")
                if verify_response.status_code == 200:
                    updated_alerts = verify_response.json()
                    for alert in updated_alerts:
                        if alert['id'] == alert_id:
                            print(f"Verified update - Status: {alert['status']}")
                            break
        else:
            print("No alerts to update")
            
            # Create a test alert manually for testing
            print("Creating a test alert manually...")
            # We'll simulate what happens in the prediction function
            alert_data = {
                "entity_id": "test_loan_12345",
                "risk_signal": "Test High Default Probability",
                "severity": "Critical",
                "prediction_score": 0.85,
                "status": "New"
            }
            
            # Since we can't directly create alerts via API, we'll make a high-risk prediction
            # that should trigger an alert
            high_risk_loan = {
                "loan_limit": "cf",
                "Gender": "Male",
                "approv_in_adv": "nopre",
                "loan_type": "type1",
                "loan_purpose": "p1",
                "Credit_Worthiness": "l1",
                "open_credit": "nopc",
                "business_or_commercial": "b/c",  # High risk
                "Neg_ammortization": "neg_ammortization",  # High risk
                "interest_only": "int_only",  # High risk
                "lump_sum_payment": "lpsm",  # High risk
                "construction_type": "sb",
                "occupancy_type": "pr",
                "Secured_by": "home",
                "total_units": "1U",
                "credit_type": "EXP",
                "co_applicant_credit_type": "CIB",
                "age": "25-34",
                "submission_of_application": "to_inst",
                "Region": "south",
                "Security_Type": "direct",
                "rate_of_interest": 20.0,  # Very high
                "Interest_rate_spread": 10.0,  # Very high
                "Upfront_charges": 15000.0,  # Very high
                "term": 360.0,
                "income": 500.0,  # Very low
                "Credit_Score": 300.0,  # Minimum
                "LTV": 99.0,  # Maximum
                "dtir1": 95.0  # Maximum
            }
            
            pred_response = session.post(f"{BASE_URL}/predict", json=high_risk_loan)
            print(f"High-risk prediction response: {pred_response.status_code}")
            if pred_response.status_code == 200:
                pred_result = pred_response.json()
                print(f"High-risk prediction result: {pred_result}")
                
                # Check alerts again
                alerts_response = session.get(f"{BASE_URL}/api/alerts")
                if alerts_response.status_code == 200:
                    new_alerts = alerts_response.json()
                    print(f"Alerts after high-risk prediction: {len(new_alerts)}")
                    for alert in new_alerts:
                        print(f"  - {alert['risk_signal']} ({alert['severity']}) - Score: {alert['prediction_score']}")

if __name__ == "__main__":
    print("Testing EWS enhancements...")
    print("=" * 50)
    
    try:
        # Login first
        if not login():
            print("Failed to login, exiting...")
            exit(1)
            
        test_macro_indicators()
        test_alerts()
        test_create_test_alert()
        test_alert_update()
        
        print("\n" + "=" * 50)
        print("Testing completed!")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running.")
    except Exception as e:
        print(f"Error during testing: {e}")