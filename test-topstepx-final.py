#!/usr/bin/env python3

import os
import sys
import json
import requests
from urllib.parse import urljoin
from datetime import datetime, timezone, timedelta

def get_jwt_token(username, api_key, base_url):
    """Get JWT token using username and API key"""
    print(f"🔐 Attempting to get JWT token for {username}...")
    
    login_url = urljoin(base_url, "/api/Auth/loginKey")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    payload = {
        "userName": username,
        "apiKey": api_key
    }
    
    try:
        response = requests.post(login_url, 
                               headers=headers, 
                               json=payload, 
                               timeout=30)
        
        print(f"   Login response: {response.status_code} {response.reason}")
        
        if response.status_code == 200:
            data = response.json()
            if 'token' in data:
                print("   ✅ JWT token obtained successfully")
                return data['token']
            else:
                print(f"   ❌ No token in response: {data}")
                return None
        else:
            print(f"   ❌ Login failed")
            try:
                error_data = response.json()
                print(f"   🔍 Error details: {error_data}")
            except:
                print(f"   🔍 Error text: {response.text[:200]}")
            return None
            
    except Exception as e:
        print(f"   ❌ Login error: {e}")
        return None

def main():
    print("🔍 TopstepX API Integration Test")
    print("=" * 50)
    
    # Get credentials from environment
    account_id = os.getenv('TOPSTEPX_ACCOUNT_ID', '297693')
    username = os.getenv('TOPSTEPX_USERNAME', 'kevinsuero072897@gmail.com') 
    api_key = os.getenv('TOPSTEPX_API_KEY')
    jwt_token = os.getenv('TOPSTEPX_JWT')
    
    if not api_key:
        print("❌ TOPSTEPX_API_KEY environment variable not set")
        return 1
        
    print(f"Account ID: {account_id}")
    print(f"Username: {username}")
    print(f"API Key: {api_key[:10]}...{api_key[-10:]}")
    
    base_url = "https://api.topstepx.com"
    
    # Get JWT token
    if not jwt_token:
        jwt_token = get_jwt_token(username, api_key, base_url)
        if not jwt_token:
            print("❌ Failed to obtain JWT token")
            return 1
    
    # Mask token for display
    if len(jwt_token) > 20:
        masked_token = f"{jwt_token[:10]}...{jwt_token[-10:]}"
    else:
        masked_token = "***masked***"
    print(f"JWT Token: {masked_token}")
    
    # Test endpoints with proper request bodies
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    print(f"\n🌐 Testing TopstepX API endpoints")
    print("=" * 50)
    
    success_count = 0
    
    # Test Contract/available with proper POST request
    print(f"\n📡 Testing: /api/Contract/available")
    try:
        url = urljoin(base_url, "/api/Contract/available")
        payload = {"live": False}  # Use simulation contracts as specified
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        print(f"   Status: {response.status_code} {response.reason}")
        
        if response.status_code == 200:
            success_count += 1
            data = response.json()
            print(f"   ✅ Success - Available contracts found")
            
            if isinstance(data, dict) and 'data' in data:
                contracts = data['data']
                print(f"   📊 Total contracts: {len(contracts)}")
                
                # Show some contract details
                for i, contract in enumerate(contracts[:3]):  # Show first 3
                    if isinstance(contract, dict):
                        symbol = contract.get('symbol', 'Unknown')
                        name = contract.get('name', 'Unknown')
                        print(f"   📈 Contract {i+1}: {symbol} - {name}")
                        
            print(f"   🎯 This is the key endpoint for live trading!")
                        
        else:
            try:
                error_data = response.json()
                print(f"   ❌ Error: {error_data}")
            except:
                print(f"   ❌ Error text: {response.text[:200]}")
                
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Test Trade search with proper request body
    print(f"\n📡 Testing: /api/Trade/search")
    try:
        url = urljoin(base_url, "/api/Trade/search")
        
        # Search for recent trades
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=30)
        
        payload = {
            "accountId": int(account_id),
            "startTimestamp": start_date.isoformat(),
            "endTimestamp": end_date.isoformat()
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        print(f"   Status: {response.status_code} {response.reason}")
        
        if response.status_code == 200:
            success_count += 1
            data = response.json()
            print(f"   ✅ Success - Trade history accessible")
            
            if isinstance(data, dict) and 'data' in data:
                trades = data['data']
                print(f"   📊 Trades found: {len(trades)}")
                
        else:
            try:
                error_data = response.json()
                print(f"   ❌ Error: {error_data}")
            except:
                print(f"   ❌ Error text: {response.text[:200]}")
                
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Test Order search
    print(f"\n📡 Testing: /api/Order/search")
    try:
        url = urljoin(base_url, "/api/Order/search")
        
        # Search for recent orders
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=30)
        
        payload = {
            "accountId": int(account_id),
            "startTimestamp": start_date.isoformat(),
            "endTimestamp": end_date.isoformat()
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        print(f"   Status: {response.status_code} {response.reason}")
        
        if response.status_code == 200:
            success_count += 1
            data = response.json()
            print(f"   ✅ Success - Order history accessible")
            
            if isinstance(data, dict) and 'data' in data:
                orders = data['data']
                print(f"   📊 Orders found: {len(orders)}")
                
        else:
            try:
                error_data = response.json()
                print(f"   ❌ Error: {error_data}")
            except:
                print(f"   ❌ Error text: {response.text[:200]}")
                
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"🏁 API test complete: {success_count}/3 key endpoints successful")
    
    if success_count > 0:
        print("✅ TopstepX API integration is working!")
        print("🎯 Your bot can connect to live TopstepX servers")
        print("📈 Ready for paper trading with simulated contracts")
    else:
        print("❌ API integration issues detected")
    
    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
