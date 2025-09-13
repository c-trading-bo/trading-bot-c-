#!/usr/bin/env python3

from dotenv import load_dotenv
import os
import requests
import json

load_dotenv()

jwt_token = os.getenv('TOPSTEPX_JWT')
base_url = os.getenv('TOPSTEPX_API_BASE', 'https://api.topstepx.com')
account_id = os.getenv('TOPSTEPX_ACCOUNT_ID', '11011203')

headers = {
    'Authorization': f'Bearer {jwt_token}',
    'Content-Type': 'application/json'
}

print("🔍 Testing Account API Endpoints - Comprehensive Research")
print("=" * 70)

def test_endpoint(method, endpoint, payload=None, description=""):
    """Test an API endpoint with comprehensive error analysis"""
    print(f"\n📡 Testing: {method} {endpoint}")
    if description:
        print(f"   📝 {description}")
    
    try:
        if method.upper() == 'GET':
            response = requests.get(f'{base_url}{endpoint}', headers=headers, timeout=30)
        elif method.upper() == 'POST':
            response = requests.post(f'{base_url}{endpoint}', headers=headers, json=payload, timeout=30)
        elif method.upper() == 'PUT':
            response = requests.put(f'{base_url}{endpoint}', headers=headers, json=payload, timeout=30)
        else:
            print(f"   ❌ Unsupported method: {method}")
            return False
            
        print(f"   📊 Status: {response.status_code} {response.reason}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"   ✅ SUCCESS!")
                if isinstance(data, dict):
                    print(f"   📋 Response keys: {list(data.keys())}")
                return True
            except json.JSONDecodeError:
                print(f"   ✅ SUCCESS (non-JSON response)")
                return True
        elif response.status_code == 404:
            print(f"   ❌ ENDPOINT NOT FOUND")
        elif response.status_code == 401:
            print(f"   ❌ UNAUTHORIZED - Check JWT token")
        elif response.status_code == 403:
            print(f"   ❌ FORBIDDEN - Check permissions")
        elif response.status_code == 400:
            try:
                error_data = response.json()
                print(f"   ⚠️  BAD REQUEST")
                print(f"   📄 Error: {json.dumps(error_data, indent=4)}")
            except:
                print(f"   ⚠️  BAD REQUEST: {response.text}")
        else:
            print(f"   ❌ FAILED")
            try:
                error_data = response.json()
                print(f"   📄 Error: {json.dumps(error_data, indent=4)}")
            except:
                print(f"   📄 Response: {response.text[:200]}")
        return False
            
    except Exception as e:
        print(f"   💥 EXCEPTION: {e}")
        return False

# Test various Account endpoint patterns
account_patterns = [
    # Standard REST patterns
    (f'/api/Account/{account_id}', 'GET', None, 'Standard REST pattern'),
    (f'/api/Account', 'GET', None, 'Get all accounts'),
    (f'/api/Account', 'POST', {"accountId": account_id}, 'POST with accountId'),
    (f'/api/Account/details', 'GET', None, 'Details endpoint'),
    (f'/api/Account/details/{account_id}', 'GET', None, 'Details with ID'),
    (f'/api/Account/info', 'GET', None, 'Info endpoint'),
    (f'/api/Account/info/{account_id}', 'GET', None, 'Info with ID'),
    
    # Query parameter patterns
    (f'/api/Account?accountId={account_id}', 'GET', None, 'Query parameter'),
    (f'/api/Account?id={account_id}', 'GET', None, 'Query with id param'),
    
    # Alternative naming patterns
    (f'/api/Accounts/{account_id}', 'GET', None, 'Plural accounts'),
    (f'/api/Accounts', 'GET', None, 'Plural accounts list'),
    (f'/api/User/Account/{account_id}', 'GET', None, 'User account pattern'),
    (f'/api/User/Accounts', 'GET', None, 'User accounts list'),
    
    # Search patterns
    (f'/api/Account/search', 'POST', {"accountId": account_id}, 'Search pattern'),
    (f'/api/Account/lookup', 'POST', {"accountId": account_id}, 'Lookup pattern'),
    (f'/api/Account/get', 'POST', {"accountId": account_id}, 'Get pattern'),
    
    # Alternative ID formats
    (f'/api/Account/{int(account_id)}', 'GET', None, 'Integer ID'),
    (f'/api/Account', 'POST', {"id": int(account_id)}, 'POST with integer ID'),
]

print(f"🎯 Testing {len(account_patterns)} Account endpoint patterns...")
successful_patterns = []

for endpoint, method, payload, description in account_patterns:
    if test_endpoint(method, endpoint, payload, description):
        successful_patterns.append((method, endpoint, payload, description))

# Test Position endpoint patterns
print(f"\n\n🔍 Testing Position API Endpoints")
print("=" * 50)

position_patterns = [
    (f'/api/Position/{account_id}', 'GET', None, 'GET with account ID'),
    (f'/api/Position', 'GET', None, 'GET all positions'),
    (f'/api/Position/search', 'POST', {"accountId": int(account_id)}, 'Search with integer ID'),
    (f'/api/Position/search', 'POST', {"accountId": account_id}, 'Search with string ID'),
    (f'/api/Position/list', 'POST', {"accountId": int(account_id)}, 'List pattern'),
    (f'/api/Position/current', 'POST', {"accountId": int(account_id)}, 'Current positions'),
    (f'/api/Positions/{account_id}', 'GET', None, 'Plural positions'),
    (f'/api/Positions', 'POST', {"accountId": int(account_id)}, 'Plural with POST'),
    (f'/api/Portfolio/{account_id}', 'GET', None, 'Portfolio pattern'),
    (f'/api/Portfolio/positions', 'POST', {"accountId": int(account_id)}, 'Portfolio positions'),
]

for endpoint, method, payload, description in position_patterns:
    if test_endpoint(method, endpoint, payload, description):
        successful_patterns.append((method, endpoint, payload, description))

# Summary
print(f"\n\n🏁 SUMMARY")
print("=" * 50)
if successful_patterns:
    print(f"✅ Found {len(successful_patterns)} working endpoints:")
    for method, endpoint, payload, description in successful_patterns:
        print(f"   {method} {endpoint} - {description}")
else:
    print("❌ No working Account/Position endpoints found")
    print("💡 Possible reasons:")
    print("   - Endpoints may require different authentication")
    print("   - API structure may be different than expected")
    print("   - Account may need activation or different permissions")
    print("   - API documentation may be needed for correct paths")

print(f"\n✅ Investigation complete!")