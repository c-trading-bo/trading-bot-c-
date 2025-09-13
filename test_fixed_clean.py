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
    
    # Test cases with proper request bodies for ES/NQ only
    test_cases = [
        {
            "name": "Contract Search (ES)",
            "endpoint": "/api/Contract/search",
            "method": "POST",
            "payload": {
                "searchText": "ES",
                "live": False
            }
        },
        {
            "name": "Contract Search (NQ)", 
            "endpoint": "/api/Contract/search",
            "method": "POST",
            "payload": {
                "searchText": "NQ",
                "live": False
            }
        },
        {
            "name": "Order Search (Last 7 days)",
            "endpoint": "/api/Order/search", 
            "method": "POST",
            "payload": {
                "accountId": account_id,
                "startTimestamp": (datetime.now() - timedelta(days=7)).isoformat() + "Z",
                "endTimestamp": datetime.now().isoformat() + "Z"
            }
        },
        {
            "name": "Trade Search (Last 7 days)",
            "endpoint": "/api/Trade/search",
            "method": "POST", 
            "payload": {
                "accountId": account_id,
                "startTimestamp": (datetime.now() - timedelta(days=7)).isoformat() + "Z",
                "endTimestamp": datetime.now().isoformat() + "Z"
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
                "accountId": account_id
            }
        }
    ]
    
    for test_case in test_cases:
        total_calls += 1
        print(f"📡 Testing: {test_case['name']} - {test_case['endpoint']}")
        
        try:
            if test_case['method'] == 'POST':
                response = requests.post(
                    f"{base_url}{test_case['endpoint']}", 
                    headers=headers,
                    json=test_case['payload'],
                    timeout=30
                )
            else:
                response = requests.get(
                    f"{base_url}{test_case['endpoint']}", 
                    headers=headers,
                    timeout=30
                )
            
            print(f"   Status: {response.status_code} {response.reason}")
            
            if response.status_code == 200:
                successful_calls += 1
                try:
                    data = response.json()
                    print(f"   ✅ Success! Response size: {len(str(data))} chars")
                    
                    # Show sample data for contract searches
                    if 'Contract' in test_case['name'] and 'data' in data:
                        contracts = data.get('data', [])
                        if contracts:
                            print(f"   📊 Found {len(contracts)} contracts")
                            for contract in contracts[:2]:  # Show first 2
                                symbol = contract.get('symbol', 'Unknown')
                                contract_id = contract.get('contractId', contract.get('id', 'N/A'))
                                print(f"      • {symbol}: {contract_id}")
                        
                except:
                    print(f"   ✅ Success! (Non-JSON response)")
                    
            elif response.status_code == 400:
                try:
                    error_data = response.json()
                    print(f"   ❌ Bad Request")
                    if 'errors' in error_data:
                        for field, messages in error_data['errors'].items():
                            print(f"      • {field}: {messages}")
                    elif 'error' in error_data:
                        print(f"      • Error: {error_data['error']}")
                except:
                    print(f"   ❌ Bad Request: {response.text[:200]}")
                    
            else:
                print(f"   ❌ Failed: {response.status_code}")
                if response.text:
                    print(f"      Response: {response.text[:200]}")
                
        except requests.exceptions.Timeout:
            print(f"   ⏰ Timeout")
        except requests.exceptions.RequestException as e:
            print(f"   🔌 Connection error: {e}")
        except Exception as e:
            print(f"   💥 Error: {e}")
            
        print()
    
    print("=" * 50)
    print(f"🏁 API test complete: {successful_calls}/{total_calls} endpoints successful")
    
    if successful_calls > 0:
        print(f"✅ {successful_calls} working endpoints found!")
        return True
    else:
        print("❌ No successful API calls - check credentials and permissions")
        return False

def test_historical_backtest_readiness():
    """Test if historical backtesting is ready"""
    
    print("\n🎯 Testing Historical Backtesting Readiness...")
    
    # Check if historical data exists
    import pathlib
    data_dir = pathlib.Path("data/historical")
    
    if not data_dir.exists():
        print("❌ Historical data directory missing")
        return False
    
    es_file = data_dir / "ES_bars.json"
    nq_file = data_dir / "NQ_bars.json"
    
    data_ready = 0
    es_bars = 0
    nq_bars = 0
    
    for symbol_file in [es_file, nq_file]:
        if symbol_file.exists():
            try:
                with open(symbol_file) as f:
                    data = json.load(f)
                bars_count = len(data)
                print(f"✅ {symbol_file.name}: {bars_count} bars")
                if symbol_file.name.startswith('ES'):
                    es_bars = bars_count
                elif symbol_file.name.startswith('NQ'):
                    nq_bars = bars_count
                data_ready += 1
            except Exception as e:
                print(f"❌ {symbol_file.name}: Error reading - {e}")
        else:
            print(f"❌ {symbol_file.name}: Missing")
    
    # Check strategy config
    strategy_file = pathlib.Path("src/BotCore/Strategy/S3-StrategyConfig.json")
    if strategy_file.exists():
        try:
            with open(strategy_file) as f:
                config = json.load(f)
            s3_enabled = False
            for strategy in config.get('Strategies', []):
                if strategy.get('id') == 'S3' and strategy.get('enabled'):
                    s3_enabled = True
                    break
            
            if s3_enabled:
                print(f"✅ S3 Strategy: Enabled and configured")
                data_ready += 1
            else:
                print(f"❌ S3 Strategy: Not enabled")
        except Exception as e:
            print(f"❌ S3 Strategy: Error reading config - {e}")
    else:
        print(f"❌ S3 Strategy: Config file missing")
    
    print(f"\n🎯 Backtesting Readiness: {data_ready}/3 components ready")
    print(f"📊 Data Summary: ES={es_bars} bars, NQ={nq_bars} bars")
    
    if data_ready >= 2 and (es_bars > 0 or nq_bars > 0):
        print("✅ Ready for historical backtesting!")
        return True
    else:
        print("❌ Historical backtesting not ready")
        return False

if __name__ == "__main__":
    print("🚀 TopstepX API & Backtesting Readiness Test")
    print("=" * 50)
    
    api_success = test_topstepx_apis_fixed()
    backtest_ready = test_historical_backtest_readiness()
    
    print("\n" + "=" * 50)
    print("📊 FINAL SUMMARY")
    print(f"🔌 API Status: {'✅ Working' if api_success else '❌ Issues'}")
    print(f"📈 Backtest Ready: {'✅ Yes' if backtest_ready else '❌ No'}")
    
    if backtest_ready:
        print("\n🎯 Next Steps - Run Historical Backtest:")
        print("1. dotnet run --project src/ML/HistoricalTrainer")
        print("2. python ml/train_monthly.py")
        print("3. python ml/WalkForwardBacktester.py")
        print("\n📈 Watch your S3 strategy practice on historical data!")
    else:
        print("\n⚠️  Setup Required:")
        print("1. Generate historical data: python generate_historical_data.py")
        print("2. Check S3 strategy enabled in config")
        print("3. Re-run this test")