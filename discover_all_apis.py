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

print("🔍 Discovering ALL Available TopstepX API Endpoints")
print("=" * 60)

def test_endpoint(method, endpoint, payload=None, expected_status=[200]):
    """Test endpoint and return True if successful"""
    try:
        if method.upper() == 'GET':
            response = requests.get(f'{base_url}{endpoint}', headers=headers, timeout=10)
        elif method.upper() == 'POST':
            response = requests.post(f'{base_url}{endpoint}', headers=headers, json=payload, timeout=10)
        
        if response.status_code in expected_status:
            return True, response.status_code, response
        else:
            return False, response.status_code, response
    except Exception as e:
        return False, 0, str(e)

# Test all possible API patterns systematically
api_categories = {
    'Account': [
        ('/api/Account/search', 'POST', {"accountId": account_id}),
        ('/api/Account/details', 'POST', {"accountId": account_id}),
        ('/api/Account/balance', 'POST', {"accountId": account_id}),
        ('/api/Account/info', 'POST', {"accountId": account_id}),
    ],
    'Position': [
        ('/api/Position/search', 'POST', {"accountId": int(account_id)}),
        ('/api/Position/current', 'POST', {"accountId": int(account_id)}),
        ('/api/Position/list', 'POST', {"accountId": int(account_id)}),
        ('/api/Position/summary', 'POST', {"accountId": int(account_id)}),
        ('/api/Positions/search', 'POST', {"accountId": int(account_id)}),
        ('/api/Portfolio/search', 'POST', {"accountId": int(account_id)}),
        ('/api/Portfolio/positions', 'POST', {"accountId": int(account_id)}),
    ],
    'Market': [
        ('/api/Market/data', 'GET', None),
        ('/api/Market/quotes', 'GET', None),
        ('/api/Market/depth', 'GET', None),
        ('/api/Market/status', 'GET', None),
    ],
    'User': [
        ('/api/User/profile', 'GET', None),
        ('/api/User/info', 'GET', None),
        ('/api/User/accounts', 'GET', None),
        ('/api/User/settings', 'GET', None),
    ],
    'System': [
        ('/api/System/status', 'GET', None),
        ('/api/System/health', 'GET', None),
        ('/api/System/version', 'GET', None),
        ('/api/Status', 'GET', None),
        ('/health', 'GET', None),
        ('/healthz', 'GET', None),
    ]
}

working_endpoints = []
total_tested = 0

for category, endpoints in api_categories.items():
    print(f"\n📊 Testing {category} APIs:")
    category_success = 0
    
    for endpoint, method, payload in endpoints:
        total_tested += 1
        success, status_code, response = test_endpoint(method, endpoint, payload)
        
        if success:
            print(f"   ✅ {method} {endpoint} - {status_code}")
            working_endpoints.append((category, method, endpoint, payload, status_code))
            category_success += 1
        elif status_code == 400:
            # 400 might mean endpoint exists but needs different payload
            print(f"   ⚠️  {method} {endpoint} - {status_code} (exists, needs correct payload)")
            try:
                error_data = response.json()
                if 'errors' in error_data:
                    print(f"      📋 Required fields: {list(error_data['errors'].keys())}")
            except:
                pass
        elif status_code in [401, 403]:
            print(f"   🔐 {method} {endpoint} - {status_code} (auth issue)")
        elif status_code == 404:
            print(f"   ❌ {method} {endpoint} - {status_code}")
        else:
            print(f"   ❓ {method} {endpoint} - {status_code}")
    
    print(f"   📈 {category}: {category_success}/{len(endpoints)} working")

# Now test the working Account/search to get more account info
print(f"\n\n🎯 Account Details Analysis:")
success, status_code, response = test_endpoint('POST', '/api/Account/search', {"accountId": account_id})
if success:
    data = response.json()
    accounts = data.get('accounts', [])
    print(f"Found {len(accounts)} accounts:")
    
    for account in accounts:
        account_id_found = account.get('id')
        name = account.get('name')
        balance = account.get('balance')
        can_trade = account.get('canTrade')
        is_visible = account.get('isVisible')
        simulated = account.get('simulated')
        
        print(f"  💼 Account {account_id_found}: {name}")
        print(f"     💰 Balance: ${balance:,.2f}")
        print(f"     📈 Can Trade: {can_trade}")
        print(f"     👁️  Visible: {is_visible}")
        print(f"     🎮 Simulated: {simulated}")
        
        # Test Position search with this account
        print(f"     🔍 Testing positions for account {account_id_found}...")
        pos_success, pos_status, pos_response = test_endpoint('POST', '/api/Position/search', {"accountId": account_id_found})
        if pos_success:
            print(f"     ✅ Position search works!")
            try:
                pos_data = pos_response.json()
                print(f"     📊 Position data keys: {list(pos_data.keys())}")
            except:
                pass
        else:
            print(f"     ❌ Position search failed: {pos_status}")

print(f"\n\n🏁 FINAL SUMMARY")
print("=" * 50)
print(f"📊 Total endpoints tested: {total_tested}")
print(f"✅ Working endpoints: {len(working_endpoints)}")

if working_endpoints:
    print(f"\n🎯 Working API Endpoints:")
    for category, method, endpoint, payload, status_code in working_endpoints:
        payload_desc = "no payload" if payload is None else f"payload: {payload}"
        print(f"   {method} {endpoint} ({payload_desc})")

print(f"\n✅ API Discovery Complete!")