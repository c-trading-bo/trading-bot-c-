#!/usr/bin/env python3

from dotenv import load_dotenv
import os
import requests
import json
from datetime import datetime, timedelta

load_dotenv()

jwt_token = os.getenv('TOPSTEPX_JWT')
base_url = os.getenv('TOPSTEPX_API_BASE', 'https://api.topstepx.com')
account_id = int(os.getenv('TOPSTEPX_ACCOUNT_ID', '11011203'))

headers = {
    'Authorization': f'Bearer {jwt_token}',
    'Content-Type': 'application/json'
}

print("🔍 Researching Position Data Alternatives")
print("=" * 50)

# Get account that can trade
print("1️⃣ Getting tradeable accounts...")
response = requests.post(f'{base_url}/api/Account/search', 
                        headers=headers, 
                        json={'accountId': str(account_id)}, 
                        timeout=30)

if response.status_code == 200:
    data = response.json()
    accounts = data.get('accounts', [])
    tradeable_accounts = [acc for acc in accounts if acc.get('canTrade', False)]
    
    print(f"Found {len(tradeable_accounts)} tradeable accounts:")
    for account in tradeable_accounts:
        print(f"  💼 {account['id']}: {account['name']} (${account['balance']:,.2f})")

# Check what Trade search returns - it might include position info
print(f"\n2️⃣ Analyzing Trade search response structure...")
start_time = datetime.now() - timedelta(days=30)  # Last 30 days
end_time = datetime.now()

trade_response = requests.post(f'{base_url}/api/Trade/search', 
                              headers=headers, 
                              json={
                                  'accountId': account_id,
                                  'startTimestamp': start_time.isoformat() + 'Z',
                                  'endTimestamp': end_time.isoformat() + 'Z'
                              }, 
                              timeout=30)

if trade_response.status_code == 200:
    trade_data = trade_response.json()
    print(f"✅ Trade search successful")
    print(f"📊 Response keys: {list(trade_data.keys())}")
    
    trades = trade_data.get('trades', [])
    print(f"📈 Found {len(trades)} trades")
    
    if trades:
        print(f"\n🔍 Sample trade structure:")
        sample_trade = trades[0]
        print(json.dumps(sample_trade, indent=2))
        
        # Check if trades contain position information
        position_fields = ['position', 'quantity', 'openQty', 'netQty', 'netPosition']
        found_position_fields = [field for field in position_fields if field in sample_trade]
        if found_position_fields:
            print(f"🎯 Position-related fields found: {found_position_fields}")
        else:
            print("❌ No obvious position fields in trade data")

# Check Order search response structure
print(f"\n3️⃣ Analyzing Order search response structure...")
order_response = requests.post(f'{base_url}/api/Order/search', 
                              headers=headers, 
                              json={
                                  'accountId': account_id,
                                  'startTimestamp': start_time.isoformat() + 'Z',
                                  'endTimestamp': end_time.isoformat() + 'Z'
                              }, 
                              timeout=30)

if order_response.status_code == 200:
    order_data = order_response.json()
    print(f"✅ Order search successful")
    print(f"📊 Response keys: {list(order_data.keys())}")
    
    orders = order_data.get('orders', [])
    print(f"📋 Found {len(orders)} orders")
    
    if orders:
        print(f"\n🔍 Sample order structure:")
        sample_order = orders[0]
        print(json.dumps(sample_order, indent=2))

# Test some alternative endpoint patterns for positions
print(f"\n4️⃣ Testing alternative position endpoint patterns...")

alternative_patterns = [
    # Balance/Portfolio patterns
    ('/api/Balance/search', {'accountId': account_id}),
    ('/api/Portfolio/search', {'accountId': account_id}),
    ('/api/Holdings/search', {'accountId': account_id}),
    
    # Current state patterns  
    ('/api/Account/positions', {'accountId': account_id}),
    ('/api/Account/balance', {'accountId': account_id}),
    ('/api/Account/portfolio', {'accountId': account_id}),
    
    # Risk/PnL patterns
    ('/api/Risk/search', {'accountId': account_id}),
    ('/api/PnL/search', {'accountId': account_id}),
    ('/api/Exposure/search', {'accountId': account_id}),
    
    # Real-time patterns
    ('/api/Live/positions', {'accountId': account_id}),
    ('/api/Current/positions', {'accountId': account_id}),
    
    # Try with different payload structures
    ('/api/Position/search', {'account': account_id}),
    ('/api/Position/search', {'id': account_id}),
    ('/api/Position/search', {'accountNumber': account_id}),
]

working_alternatives = []
for endpoint, payload in alternative_patterns:
    try:
        response = requests.post(f'{base_url}{endpoint}', 
                               headers=headers, 
                               json=payload, 
                               timeout=10)
        
        if response.status_code == 200:
            print(f"✅ {endpoint} - WORKS!")
            working_alternatives.append((endpoint, payload))
        elif response.status_code == 400:
            print(f"⚠️  {endpoint} - 400 (exists but wrong payload)")
            try:
                error_data = response.json()
                if 'errors' in error_data:
                    print(f"   📋 Required: {list(error_data['errors'].keys())}")
            except:
                pass
        else:
            print(f"❌ {endpoint} - {response.status_code}")
    except Exception as e:
        print(f"💥 {endpoint} - Error: {e}")

print(f"\n🏁 POSITION DATA RESEARCH SUMMARY")
print("=" * 50)

if working_alternatives:
    print(f"✅ Found {len(working_alternatives)} working position alternatives:")
    for endpoint, payload in working_alternatives:
        print(f"   POST {endpoint}")
else:
    print("❌ No direct position endpoints found")
    print("\n💡 Alternative approaches:")
    print("   1. Calculate positions from Trade/Order search results")
    print("   2. Use SignalR real-time position updates")
    print("   3. Maintain position state internally in the bot")
    print("   4. Check if Account/search returns position summary")

print(f"\n✅ Research complete!")