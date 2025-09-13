#!/usr/bin/env python3

import os
import sys
import json
import requests
from urllib.parse import urljoin
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
    print("🔍 TopstepX Authentication & Connectivity Test")
    print("=" * 50)
    
    # Get credentials from environment
    account_id = os.getenv('TOPSTEPX_ACCOUNT_ID')
    username = os.getenv('TOPSTEPX_USERNAME') 
    api_key = os.getenv('TOPSTEPX_API_KEY')
    jwt_token = os.getenv('TOPSTEPX_JWT')
    
    if not account_id:
        print("❌ TOPSTEPX_ACCOUNT_ID environment variable not set")
        return 1
        
    print(f"Account ID: {account_id}")
    print(f"Username: {username or 'Not set'}")
    print(f"API Key: {'Set (' + api_key[:10] + '...' + api_key[-10:] + ')' if api_key else 'Not set'}")
    
    base_url = "https://api.topstepx.com"
    
    # Try to get JWT token if we don't have one
    if not jwt_token and username and api_key:
        jwt_token = get_jwt_token(username, api_key, base_url)
        if not jwt_token:
            print("❌ Failed to obtain JWT token")
            return 1
    elif not jwt_token:
        print("❌ No JWT token and no credentials to obtain one")
        return 1
    
    # Mask token for display (show first and last 10 chars)
    if len(jwt_token) > 20:
        masked_token = f"{jwt_token[:10]}...{jwt_token[-10:]}"
    else:
        masked_token = "***masked***"
    print(f"JWT Token: {masked_token}")
    
    # Test endpoints
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    endpoints = [
        # Account endpoints
        f"/api/Account?accountId={account_id}",
        f"/api/Account/{account_id}",
        "/api/Account",
        
        # Position and trade endpoints 
        "/api/Position/search",
        "/api/Trade/search",
        "/api/Order/search",
        
        # Contract endpoints
        "/api/Contract/available",
        "/api/Contract/search"
    ]
    
    print(f"\n🌐 Testing connectivity to {base_url}")
    print("=" * 50)
    
    success_count = 0
    total_count = len(endpoints)
    
    for endpoint in endpoints:
        url = urljoin(base_url, endpoint)
        print(f"\n📡 Testing: {endpoint}")
        
        try:
            if "search" in endpoint.lower():
                # Use POST for search endpoints
                response = requests.post(url, 
                                       headers=headers, 
                                       json={}, 
                                       timeout=10)
            else:
                # Use GET for other endpoints
                response = requests.get(url, headers=headers, timeout=10)
            
            print(f"   Status: {response.status_code} {response.reason}")
            
            if response.status_code == 200:
                success_count += 1
                try:
                    data = response.json()
                    print(f"   ✅ Success - Response size: {len(str(data))} chars")
                    
                    # Show sample data structure
                    if isinstance(data, dict):
                        if 'data' in data:
                            print(f"   📊 Data field type: {type(data['data'])}")
                            if isinstance(data['data'], list):
                                print(f"   📊 Data array length: {len(data['data'])}")
                        print(f"   🔧 Response keys: {list(data.keys())}")
                    elif isinstance(data, list):
                        print(f"   📊 Response array length: {len(data)}")
                        
                except json.JSONDecodeError:
                    print(f"   ⚠️ Non-JSON response: {response.text[:100]}...")
                    
            elif response.status_code == 401:
                print(f"   ❌ Authentication failed")
                try:
                    error_data = response.json()
                    print(f"   🔍 Error details: {error_data}")
                except:
                    print(f"   🔍 Error text: {response.text[:200]}")
                    
            elif response.status_code == 404:
                print(f"   ❌ Endpoint not found")
                
            elif response.status_code == 403:
                print(f"   ❌ Access forbidden - may need API permissions")
                
            else:
                print(f"   ❌ Request failed")
                try:
                    error_data = response.json()
                    print(f"   🔍 Error details: {error_data}")
                except:
                    print(f"   🔍 Error text: {response.text[:200]}")
                    
        except requests.exceptions.Timeout:
            print(f"   ❌ Request timeout")
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Connection error")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print(f"🏁 Connectivity test complete: {success_count}/{total_count} endpoints successful")
    
    if success_count > 0:
        print("✅ TopstepX API is accessible and responding to requests")
    else:
        print("❌ No successful API calls - check credentials and permissions")
    
    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
