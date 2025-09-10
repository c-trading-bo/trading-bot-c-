#!/usr/bin/env python3
import requests
import os
import sys

# Load environment
jwt_token = os.getenv('TOPSTEPX_JWT', '')
account_id = os.getenv('TOPSTEPX_ACCOUNT_ID', '')

print("🔍 TopstepX Connectivity Test")
print("=" * 50)
print(f"Account ID: {account_id}")
print(f"JWT Token: {jwt_token[:50]}..." if jwt_token else "No JWT token")

if not jwt_token:
    print("❌ No JWT token found in environment")
    sys.exit(1)

# Test endpoints with correct TopstepX API format
endpoints = [
    ("Account Info (Query)", f"/api/Account?accountId={account_id}"),
    ("Account Info (Path)", f"/api/Account/{account_id}"),
    ("Account Risk", f"/api/Account/risk?accountId={account_id}"),
    ("Account PnL", f"/api/Account/pnl?accountId={account_id}&scope=today"),
    ("Positions", f"/api/Account/{account_id}/positions"),
    ("Contracts (GET)", "/api/Contract/available"),
    ("Open Positions (POST)", "/api/Position/searchOpen"),
]

headers = {
    'Authorization': f'Bearer {jwt_token}',
    'User-Agent': 'TopstepX-TradingBot-Test/1.0',
    'Content-Type': 'application/json'
}

base_url = "https://api.topstepx.com"

for name, endpoint in endpoints:
    try:
        print(f"\n🔗 Testing {name}: {base_url}{endpoint}")
        
        # Use POST for search endpoints
        if "/search" in endpoint:
            payload = {"accountId": account_id}
            response = requests.post(f"{base_url}{endpoint}", json=payload, headers=headers, timeout=10)
        else:
            response = requests.get(f"{base_url}{endpoint}", headers=headers, timeout=10)
            
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                if isinstance(data, dict):
                    print(f"   ✅ Success - Response has {len(data)} fields")
                    # Print first few keys if it's a dict
                    if data:
                        keys = list(data.keys())[:3]
                        print(f"   📋 Sample fields: {keys}")
                elif isinstance(data, list):
                    print(f"   ✅ Success - Response has {len(data)} items")
                    if data:
                        print(f"   📋 First item keys: {list(data[0].keys())[:3] if isinstance(data[0], dict) else 'Not a dict'}")
            except:
                print(f"   ✅ Success - Response: {response.text[:100]}...")
        elif response.status_code == 401:
            print("   ❌ Unauthorized - JWT token may be invalid/expired")
        elif response.status_code == 403:
            print("   ❌ Forbidden - Account may not have API permissions")
        elif response.status_code == 404:
            print("   ❌ Not Found - Endpoint may not exist or account not set up")
        else:
            print(f"   ❌ Error - {response.text[:200]}")
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Connection Error: {e}")

print("\n🎯 Summary:")
print("If you see 401/403 errors: Check account API permissions")
print("If you see 404 errors: Account may need setup or endpoint wrong")
print("If you see connection errors: Network/firewall issues")
