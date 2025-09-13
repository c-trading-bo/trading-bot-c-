#!/usr/bin/env python3

from dotenv import load_dotenv
import os
import requests
import json
from datetime import datetime, timedelta

load_dotenv()

jwt_token = os.getenv('TOPSTEPX_JWT')
base_url = os.getenv('TOPSTEPX_API_BASE', 'https://api.topstepx.com')
account_id = os.getenv('TOPSTEPX_ACCOUNT_ID', '11011203')

headers = {
    'Authorization': f'Bearer {jwt_token}',
    'Content-Type': 'application/json'
}

print("🚀 COMPREHENSIVE TopstepX API Test - ALL ENDPOINTS MUST WORK")
print("=" * 70)
print(f"🎯 Target: 100% API endpoint success rate")
print(f"🔑 JWT: {len(jwt_token)} chars")
print(f"👤 Account: {account_id}")
print("=" * 70)

def test_endpoint_comprehensive(method, endpoint, payloads, description=""):
    """Test endpoint with multiple payload variations until one works"""
    print(f"\n📡 {method} {endpoint}")
    print(f"   📝 {description}")
    
    success_count = 0
    
    for i, payload in enumerate(payloads):
        try:
            if method.upper() == 'GET':
                response = requests.get(f'{base_url}{endpoint}', headers=headers, timeout=30)
            elif method.upper() == 'POST':
                response = requests.post(f'{base_url}{endpoint}', headers=headers, json=payload, timeout=30)
            
            print(f"   🔧 Attempt {i+1}: {json.dumps(payload)[:60] if payload else 'No payload'}...")
            
            if response.status_code == 200:
                print(f"   ✅ SUCCESS! Status: {response.status_code}")
                try:
                    data = response.json()
                    print(f"   📊 Response keys: {list(data.keys())}")
                    if 'data' in data:
                        print(f"   📈 Data items: {len(data['data']) if isinstance(data['data'], list) else 'object'}")
                    return True
                except:
                    print(f"   ✅ SUCCESS! (Non-JSON response)")
                    return True
            elif response.status_code == 400:
                try:
                    error_data = response.json()
                    print(f"   ⚠️  400 BAD REQUEST")
                    if 'errors' in error_data:
                        required_fields = []
                        for field, messages in error_data['errors'].items():
                            if 'required' in str(messages).lower():
                                required_fields.append(field)
                        if required_fields:
                            print(f"   📋 Missing required: {required_fields}")
                        else:
                            print(f"   📋 Error fields: {list(error_data['errors'].keys())}")
                except:
                    print(f"   ⚠️  400 BAD REQUEST: {response.text[:100]}")
            elif response.status_code == 404:
                print(f"   ❌ 404 NOT FOUND")
            elif response.status_code == 401:
                print(f"   🔐 401 UNAUTHORIZED")
            elif response.status_code == 403:
                print(f"   🚫 403 FORBIDDEN")
            elif response.status_code == 405:
                print(f"   🚷 405 METHOD NOT ALLOWED")
            else:
                print(f"   ❓ {response.status_code}: {response.text[:100]}")
                
        except Exception as e:
            print(f"   💥 EXCEPTION: {e}")
    
    print(f"   ❌ ALL ATTEMPTS FAILED")
    return False

# Start time for searches
start_time = (datetime.now() - timedelta(days=90)).isoformat() + 'Z'
end_time = datetime.now().isoformat() + 'Z'

# Comprehensive test matrix - try EVERY possible variation
test_matrix = [
    # CONTRACT ENDPOINTS
    ('POST', '/api/Contract/available', [
        {'live': False},
        {'live': True},
        {},
        {'simulated': True},
        {'environment': 'simulation'}
    ], 'Get available contracts'),
    
    ('GET', '/api/Contract/available', [
        None
    ], 'Get contracts via GET'),
    
    ('POST', '/api/Contract/search', [
        {'searchText': 'ES', 'live': False},
        {'searchText': 'NQ', 'live': False},
        {'symbol': 'ES', 'live': False},
        {'contractId': 'CON.F.US.EP.U25', 'live': False},
        {'search': 'ES', 'live': False},
        {'query': 'ES', 'live': False}
    ], 'Search contracts'),
    
    # ACCOUNT ENDPOINTS - Found working, test variations
    ('POST', '/api/Account/search', [
        {'accountId': account_id},
        {'accountId': int(account_id)},
        {'id': account_id},
        {'id': int(account_id)},
        {}
    ], 'Search accounts'),
    
    ('GET', '/api/Account', [
        None
    ], 'Get accounts via GET'),
    
    ('POST', '/api/Account/details', [
        {'accountId': account_id},
        {'accountId': int(account_id)},
        {'id': account_id}
    ], 'Get account details'),
    
    ('POST', '/api/Account/balance', [
        {'accountId': account_id},
        {'accountId': int(account_id)}
    ], 'Get account balance'),
    
    ('POST', '/api/Account/info', [
        {'accountId': account_id},
        {'accountId': int(account_id)}
    ], 'Get account info'),
    
    # TRADE ENDPOINTS - Found working, test variations
    ('POST', '/api/Trade/search', [
        {
            'accountId': int(account_id),
            'startTimestamp': start_time,
            'endTimestamp': end_time
        },
        {
            'accountId': account_id,
            'startTimestamp': start_time,
            'endTimestamp': end_time
        },
        {
            'accountId': int(account_id),
            'from': start_time,
            'to': end_time
        },
        {
            'accountId': int(account_id),
            'dateFrom': start_time,
            'dateTo': end_time
        }
    ], 'Search trades'),
    
    ('POST', '/api/Trade/list', [
        {'accountId': int(account_id)},
        {'accountId': account_id}
    ], 'List trades'),
    
    ('POST', '/api/Trade/history', [
        {
            'accountId': int(account_id),
            'startTimestamp': start_time,
            'endTimestamp': end_time
        }
    ], 'Get trade history'),
    
    # ORDER ENDPOINTS - Found working, test variations
    ('POST', '/api/Order/search', [
        {
            'accountId': int(account_id),
            'startTimestamp': start_time,
            'endTimestamp': end_time
        },
        {
            'accountId': account_id,
            'startTimestamp': start_time,
            'endTimestamp': end_time
        },
        {
            'accountId': int(account_id),
            'from': start_time,
            'to': end_time
        }
    ], 'Search orders'),
    
    ('POST', '/api/Order/list', [
        {'accountId': int(account_id)},
        {'accountId': account_id}
    ], 'List orders'),
    
    ('POST', '/api/Order/history', [
        {
            'accountId': int(account_id),
            'startTimestamp': start_time,
            'endTimestamp': end_time
        }
    ], 'Get order history'),
    
    # POSITION ENDPOINTS - Try every variation
    ('POST', '/api/Position/search', [
        {'accountId': int(account_id)},
        {'accountId': account_id},
        {'id': int(account_id)},
        {'account': int(account_id)},
        {'accountNumber': account_id},
        {}
    ], 'Search positions'),
    
    ('GET', '/api/Position', [
        None
    ], 'Get positions via GET'),
    
    ('POST', '/api/Position/current', [
        {'accountId': int(account_id)},
        {'accountId': account_id}
    ], 'Get current positions'),
    
    ('POST', '/api/Position/list', [
        {'accountId': int(account_id)},
        {'accountId': account_id}
    ], 'List positions'),
    
    ('POST', '/api/Positions/search', [
        {'accountId': int(account_id)},
        {'accountId': account_id}
    ], 'Search positions (plural)'),
    
    ('POST', '/api/Portfolio/search', [
        {'accountId': int(account_id)},
        {'accountId': account_id}
    ], 'Search portfolio'),
    
    ('POST', '/api/Portfolio/positions', [
        {'accountId': int(account_id)},
        {'accountId': account_id}
    ], 'Get portfolio positions'),
    
    ('POST', '/api/Balance/search', [
        {'accountId': int(account_id)},
        {'accountId': account_id}
    ], 'Search balance'),
    
    # USER/PROFILE ENDPOINTS
    ('GET', '/api/User/profile', [None], 'Get user profile'),
    ('GET', '/api/User/info', [None], 'Get user info'),
    ('GET', '/api/User/accounts', [None], 'Get user accounts'),
    ('POST', '/api/User/search', [{'userId': account_id}], 'Search user'),
    
    # MARKET DATA ENDPOINTS
    ('GET', '/api/Market/status', [None], 'Get market status'),
    ('POST', '/api/Market/data', [{'symbol': 'ES'}, {'contractId': 'CON.F.US.EP.U25'}], 'Get market data'),
    ('POST', '/api/Market/quotes', [{'symbol': 'ES'}, {'contractId': 'CON.F.US.EP.U25'}], 'Get quotes'),
    
    # SYSTEM ENDPOINTS
    ('GET', '/api/System/status', [None], 'System status'),
    ('GET', '/api/System/health', [None], 'System health'),
    ('GET', '/api/Status', [None], 'Status'),
    ('GET', '/health', [None], 'Health check'),
    ('GET', '/healthz', [None], 'Health check (alt)'),
]

# Run comprehensive tests
working_endpoints = []
total_tests = len(test_matrix)

for i, (method, endpoint, payloads, description) in enumerate(test_matrix):
    print(f"\n{'='*70}")
    print(f"🧪 TEST {i+1}/{total_tests}")
    
    if test_endpoint_comprehensive(method, endpoint, payloads, description):
        working_endpoints.append((method, endpoint, description))

# FINAL RESULTS
print(f"\n\n🏁 FINAL COMPREHENSIVE RESULTS")
print("=" * 70)
print(f"📊 Total endpoint tests: {total_tests}")
print(f"✅ Working endpoints: {len(working_endpoints)}")
print(f"📈 Success rate: {(len(working_endpoints)/total_tests)*100:.1f}%")

if working_endpoints:
    print(f"\n🎯 ALL WORKING ENDPOINTS:")
    for method, endpoint, description in working_endpoints:
        print(f"   ✅ {method} {endpoint} - {description}")

target_success_rate = 80  # 80% target
actual_success_rate = (len(working_endpoints)/total_tests)*100

if actual_success_rate >= target_success_rate:
    print(f"\n🎉 SUCCESS! Achieved {actual_success_rate:.1f}% (target: {target_success_rate}%)")
else:
    print(f"\n⚠️  Target not met: {actual_success_rate:.1f}% (target: {target_success_rate}%)")
    print(f"💡 Consider additional endpoint variations or API documentation review")

print(f"\n✅ Comprehensive API testing complete!")