from fastapi.testclient import TestClient
import json
import traceback
import sys
import os
# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import main

client = TestClient(main.app)

print('Logging in as admin')
resp = client.post('/admin-login', data={'username':'admin','password':'password'})
print('Login status:', resp.status_code)

# Create customer
body = {
    "loan_limit":"nlt","Gender":"Male","approv_in_adv":"nopre","loan_type":"type1","loan_purpose":"SmokeTest","Credit_Worthiness":"good","open_credit":"nopc","business_or_purpose":"bp1","Neg_ammortization":"not_neg","interest_only":"not_int","lump_sum_payment":"not_lpsm","Occupancy_Type":"pr","Secured_by":"home","total_units":"1U","credit_type":"CIB","co_applicant_credit_type":"NA","age":"25-34","submission_of_application":"to_inst","Region":"North","Security_Type":"direct","rate_of_interest":4.5,"Interest_rate_spread":0.5,"Upfront_charges":1500.0,"term":360.0,"income":5000.0,"Credit_Score":720,"LTV":80.0,"dtir1":30.0
}

try:
    print('\nCreating customer...')
    r = client.post('/api/customers', json=body)
    print('Create status:', r.status_code)
    print('Create resp:', r.text[:1000])
    rjson = r.json()
    cid = rjson.get('id')
    print('Created ID:', cid)

    print('\nUpdating customer...')
    body_update = dict(body)
    body_update['income'] = 5500.0
    body_update['Credit_Score'] = 730
    r2 = client.put(f'/api/customers/{cid}', json=body_update)
    print('Update status:', r2.status_code)
    print('Update resp:', r2.text[:1000])

    print('\nPredicting for customer...')
    r3 = client.post(f'/api/customers/{cid}/predict')
    print('Predict status:', r3.status_code)
    if r3.status_code == 200:
        print('Predict resp:', json.dumps(r3.json(), indent=2)[:2000])
    else:
        print('Predict error:', r3.text[:1000])

    print('\nDeleting customer...')
    r4 = client.delete(f'/api/customers/{cid}')
    print('Delete status:', r4.status_code)
    print('Delete resp:', r4.text[:1000])

    print('\nVerifying deletion (search)...')
    r5 = client.get(f'/api/customers?per_page=1&page=1&search_id={cid}')
    print('Search status:', r5.status_code, 'body:', r5.text[:2000])

except Exception as e:
    print('Exception during smoke test')
    traceback.print_exc()