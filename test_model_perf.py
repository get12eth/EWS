import requests
import json

# First, let's log in to get a session cookie
login_url = "http://localhost:8000/admin-login"
perf_url = "http://localhost:8000/api/model_performance"

# Login with default credentials
login_data = {
    "username": "admin",
    "password": "password"
}

# Create a session to maintain cookies
session = requests.Session()

try:
    # Log in
    login_response = session.post(login_url, data=login_data)
    print(f"Login Status Code: {login_response.status_code}")
    
    # Check if login was successful
    if login_response.status_code == 200 or login_response.status_code == 302:
        print("Login successful")
        
        # Now try to access the model performance endpoint
        perf_response = session.get(perf_url)
        print(f"Performance Data Status Code: {perf_response.status_code}")
        
        if perf_response.status_code == 200:
            data = perf_response.json()
            print("Model Performance Data:")
            print(json.dumps(data, indent=2, default=str))
        else:
            print(f"Error getting performance data: {perf_response.text}")
    else:
        print(f"Login failed: {login_response.text}")
        
except Exception as e:
    print(f"Error: {e}")