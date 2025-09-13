#!/usr/bin/env python3
"""
Fixed TopstepX API Testing with Proper Request Bodies
Addresses validation errors by providing required fields
"""
import requests
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

def test_topstepx_apis_fixed():
    """Test TopstepX APIs with proper request bodies"""
    
    load_dotenv()
    
    jwt_token = os.getenv('TOPSTEPX_JWT')
    account_id = int(os.getenv('TOPSTEPX_ACCOUNT_ID', '11011203'))
    base_url = os.getenv('TOPSTEPX_API_BASE', 'https://api.topstepx.com')
    
    if not jwt_token:
        print("❌ No JWT token found in environment")
        return False
    
    headers = {
        'Authorization': f'Bearer {jwt_token}',
        'Content-Type': 'application/json'
    }
    
    print(f"🔧 Testing TopstepX APIs with proper request bodies...")
    print(f"🔑 Account ID: {account_id}")
    print(f"🌐 Base URL: {base_url}")
    print()
    
    successful_calls = 0
    total_calls = 0
    
    # Test cases with proper request bodies
    test_cases = [
        {
            "name": "Contract Search (ES)",
            "endpoint": "/api/Contract/search",
            "method": "POST",
            "payload": {
                "request": {
                    "searchText": "ES",
                    "live": False
                }
            }
        },
        {
            "name": "Contract Search (NQ)", 
            "endpoint": "/api/Contract/search",
            "method": "POST",
            "payload": {
                "request": {
                    "searchText": "NQ",
                    "live": False
                }
            }
        },
        {
            "name": "Order Search",
            "endpoint": "/api/Order/search", 
            "method": "POST",
            "payload": {
                "request": {
                    "accountId": account_id,
                    "startTimestamp": (datetime.utcnow() - timedelta(days=7)).isoformat() + "Z",
                    "endTimestamp": datetime.utcnow().isoformat() + "Z"
                }
            }
        },
        {
            "name": "Trade Search",
            "endpoint": "/api/Trade/search",
            "method": "POST", 
            "payload": {
                "request": {
                    "accountId": account_id,
                    "startTimestamp": (datetime.utcnow() - timedelta(days=7)).isoformat() + "Z",
                    "endTimestamp": datetime.utcnow().isoformat() + "Z"
                }
            }
        },
        {
            "name": "Contract Available (SIM)",
            "endpoint": "/api/Contract/available",
            "method": "POST",
            "payload": {
                "live": False
            }
        },
        {
            "name": "Account Search",
            "endpoint": "/api/Account/search",
            "method": "POST",
            "payload": {
                "request": {
                    "accountId": account_id
                }
            }
        }
    ]
print("=" * 60)
print(f"🎯 Base URL: {base_url}")
print(f"👤 Account ID: {account_id}")
print(f"🔑 JWT Length: {len(jwt_token) if jwt_token else 0} chars")
print("=" * 60)

def test_endpoint(method, endpoint, payload=None, description=""):
    """Test an API endpoint with proper error handling"""
    print(f"\n📡 Testing: {endpoint}")
    if description:
        print(f"   Description: {description}")
    
    try:
        if method.upper() == 'GET':
            response = requests.get(f'{base_url}{endpoint}', headers=headers, timeout=30)
        elif method.upper() == 'POST':
            response = requests.post(f'{base_url}{endpoint}', headers=headers, json=payload, timeout=30)
        else:
            print(f"   ❌ Unsupported method: {method}")
            return False
            
        print(f"   Status: {response.status_code} {response.reason}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"   ✅ Success!")
                if isinstance(data, dict):
                    if 'data' in data:
                        print(f"   📊 Data items: {len(data['data']) if isinstance(data['data'], list) else 'object'}")
                    elif 'contracts' in data:
                        print(f"   📊 Contracts: {len(data['contracts']) if isinstance(data['contracts'], list) else 'object'}")
                    else:
                        print(f"   📊 Response keys: {list(data.keys())[:5]}")  # Show first 5 keys
                return True
            except json.JSONDecodeError:
                print(f"   ✅ Success (non-JSON response)")
                return True
        else:
            print(f"   ❌ Request failed")
            try:
                error_data = response.json()
                print(f"   🔍 Error details: {error_data}")
            except:
                print(f"   🔍 Error text: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"   ❌ Request timeout")
        return False
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Connection error")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

# Test endpoints with proper payloads
successful_tests = 0
total_tests = 0

# 1. Contract endpoints
total_tests += 1
if test_endpoint('GET', '/api/Contract/available', description="Get available contracts (GET)"):
    successful_tests += 1

total_tests += 1
if test_endpoint('POST', '/api/Contract/available', {"live": False}, "Get sim contracts (POST)"):
    successful_tests += 1

total_tests += 1
if test_endpoint('POST', '/api/Contract/search', {
    "searchText": "ES",
    "live": False
}, "Search for ES contracts"):
    successful_tests += 1

# 2. Account endpoints (try different variations)
account_endpoints = [
    f'/api/Account/{account_id}',
    f'/api/Account?accountId={account_id}',
    '/api/Account'
]

for endpoint in account_endpoints:
    total_tests += 1
    if test_endpoint('GET', endpoint, description="Get account info"):
        successful_tests += 1
        break  # Stop on first successful account endpoint

# 3. Position endpoint
total_tests += 1
if test_endpoint('POST', '/api/Position/search', {
    "accountId": int(account_id) if account_id.isdigit() else account_id
}, "Search positions"):
    successful_tests += 1

# 4. Trade search with proper timestamp
start_time = datetime.utcnow() - timedelta(days=7)  # Last 7 days
end_time = datetime.utcnow()

total_tests += 1
if test_endpoint('POST', '/api/Trade/search', {
    "accountId": int(account_id) if account_id.isdigit() else account_id,
    "startTimestamp": start_time.isoformat() + "Z",
    "endTimestamp": end_time.isoformat() + "Z"
}, "Search trades (last 7 days)"):
    successful_tests += 1

# 5. Order search with proper timestamp
total_tests += 1
if test_endpoint('POST', '/api/Order/search', {
    "accountId": int(account_id) if account_id.isdigit() else account_id,
    "startTimestamp": start_time.isoformat() + "Z",
    "endTimestamp": end_time.isoformat() + "Z"
}, "Search orders (last 7 days)"):
    successful_tests += 1

# Summary
print("\n" + "=" * 60)
print(f"🏁 Connectivity test complete: {successful_tests}/{total_tests} endpoints successful")

if successful_tests == 0:
    print("❌ No successful API calls - check credentials and permissions")
elif successful_tests == total_tests:
    print("✅ All API endpoints working correctly!")
else:
    print(f"⚠️  Partial success - {total_tests - successful_tests} endpoints need attention")

# If Contract/available worked, try to get contract details
if successful_tests > 0:
    print("\n🔍 Attempting to get contract details...")
    try:
        # Try to get contracts
        response = requests.post(f'{base_url}/api/Contract/available', 
                               headers=headers, 
                               json={"live": False}, 
                               timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            contracts = data.get('contracts', data.get('data', []))
            
            print(f"📊 Found {len(contracts)} contracts")
            
            # Look for ES/NQ contracts
            es_contracts = []
            nq_contracts = []
            
            for contract in contracts:
                if isinstance(contract, dict):
                    symbol = str(contract.get('symbol', ''))
                    contract_id = contract.get('id', contract.get('contractId', 'N/A'))
                    
                    if 'ES' in symbol:
                        es_contracts.append((symbol, contract_id))
                    elif 'NQ' in symbol or 'MNQ' in symbol:
                        nq_contracts.append((symbol, contract_id))
            
            if es_contracts:
                print(f"\n📈 ES contracts found: {len(es_contracts)}")
                for symbol, contract_id in es_contracts[:3]:  # Show first 3
                    print(f"   {symbol}: {contract_id}")
                    
            if nq_contracts:
                print(f"\n📈 NQ/MNQ contracts found: {len(nq_contracts)}")
                for symbol, contract_id in nq_contracts[:3]:  # Show first 3
                    print(f"   {symbol}: {contract_id}")
                    
    except Exception as e:
        print(f"❌ Error getting contract details: {e}")

print("\n✅ API test completed!")