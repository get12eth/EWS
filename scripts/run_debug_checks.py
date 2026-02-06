import os
import sys
# Ensure the project root is on sys.path (so we can import main)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ensure DEBUG_MODE is enabled before importing main
os.environ['DEBUG_MODE'] = '1'
# Optionally set DATABASE_URL here if you want to test MySQL; otherwise it will use default
# os.environ['DATABASE_URL'] = "mysql+pymysql://root:Bant%406963@127.0.0.1:3306/lon-default"

import importlib
import main
importlib.reload(main)
from fastapi.testclient import TestClient

client = TestClient(main.app)

print('DEBUG_MODE in main:', main.DEBUG_MODE)

# Perform admin login to obtain session cookie for protected endpoints
print('\nLogging in as admin...')
login_resp = client.post('/admin-login', data={'username': main.ADMIN_USERNAME, 'password': main.ADMIN_PASSWORD})
print('Login status:', login_resp.status_code)
try:
    print('Login headers:', dict(login_resp.headers))
except Exception:
    pass

for path in ['/api/debug/db_info', '/api/debug/customers/count']:
    r = client.get(path)
    print('\nGET', path)
    print('Status:', r.status_code)
    try:
        print('JSON:', r.json())
    except Exception:
        print('Text:', r.text[:1000])

print('\nPOST /api/debug/customers/create_test')
r = client.post('/api/debug/customers/create_test')
print('Status:', r.status_code)
try:
    print('JSON:', r.json())
except Exception:
    print('Text:', r.text[:1000])

# Also list first 3 customers
r = client.get('/api/customers?per_page=10&page=1')
print('\nGET /api/customers?per_page=10&page=1')
print('Status:', r.status_code)
try:
    print('JSON:', r.json())
except Exception:
    print('Text:', r.text[:1000])
