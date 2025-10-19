"""
Direct REST API test - bypass SDK completely
Try to get market data using just HTTP requests
"""
import requests
import os
from dotenv import load_dotenv
import json

load_dotenv()

api_key = os.getenv('PROJECT_X_API_KEY')
username = os.getenv('PROJECT_X_USERNAME')

print("Testing TopstepX REST API directly (no SDK)")
print("=" * 60)

# Try different quote/market data endpoints
endpoints_to_try = [
    ('GET', 'https://api.topstepx.com/api/market/quotes/ES'),
    ('GET', 'https://api.topstepx.com/api/market/quotes/NQ'),
    ('GET', 'https://api.topstepx.com/api/quotes/ES'),
    ('GET', 'https://api.topstepx.com/api/v1/market/ES'),
    ('GET', 'https://api.topstepx.com/market/ES'),
]

# Try different auth header formats
auth_formats = [
    {'X-API-Key': api_key, 'X-Username': username},
    {'Authorization': f'Bearer {api_key}'},
    {'Authorization': f'ApiKey {api_key}'},
    {'X-API-Key': api_key},
]

print(f"API Key: {api_key[:15]}...")
print(f"Username: {username}\n")

found_working = False

for method, url in endpoints_to_try:
    for auth_headers in auth_formats:
        try:
            response = requests.request(
                method,
                url,
                headers={**auth_headers, 'Content-Type': 'application/json'},
                timeout=5
            )
            
            if response.status_code == 200:
                print(f"✅ SUCCESS: {method} {url}")
                print(f"   Headers: {list(auth_headers.keys())}")
                print(f"   Response: {response.text[:200]}")
                found_working = True
                break
            elif response.status_code != 404:
                print(f"⚠️  {response.status_code}: {method} {url}")
                
        except Exception as e:
            pass
    
    if found_working:
        break

if not found_working:
    print("\n❌ No working REST endpoints found")
    print("   This confirms authentication is required and current key is not working")
    print("\n💡 The API key needs to be activated in TopstepX portal")
