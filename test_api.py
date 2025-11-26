import requests
import json

#Test the model performance API endpoint
url = "http://localhost:8000/api/model_performance"



try:
    response = requests.get(url)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("API Response:")
        print(json.dumps(data, indent=2))
    else:
        print(f"Error: {response.text}")
except Exception as e:

    
    print(f"Failed to connect to API: {e}")
    print("Make sure the server is running on http://localhost:8000")