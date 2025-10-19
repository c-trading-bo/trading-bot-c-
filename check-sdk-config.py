"""
Check what environment variables the SDK needs and what it's getting
"""
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("Environment Variables Check")
print("=" * 70)

# What the SDK looks for (from project_x_py source)
sdk_vars = [
    'PROJECT_X_API_KEY',
    'PROJECT_X_USERNAME', 
    'PROJECT_X_ACCOUNT_NAME',
    'PROJECT_X_BASE_URL',
    'PROJECTX_API_URL',
    'PROJECTX_USER_HUB_URL',
    'PROJECTX_MARKET_HUB_URL',
]

# Also check legacy vars
legacy_vars = [
    'TOPSTEPX_API_KEY',
    'TOPSTEPX_USERNAME',
    'TOPSTEPX_ACCOUNT_ID',
    'TOPSTEPX_API_BASE',
]

print("\n📦 SDK Variables (PROJECT_X_*):")
print("-" * 70)
for var in sdk_vars:
    value = os.getenv(var)
    if value:
        if 'KEY' in var:
            print(f"✅ {var}: {value[:15]}...")
        else:
            print(f"✅ {var}: {value}")
    else:
        print(f"❌ {var}: NOT SET")

print("\n📦 Legacy Variables (TOPSTEPX_*):")
print("-" * 70)
for var in legacy_vars:
    value = os.getenv(var)
    if value:
        if 'KEY' in var:
            print(f"✅ {var}: {value[:15]}...")
        else:
            print(f"✅ {var}: {value}")
    else:
        print(f"❌ {var}: NOT SET")

print("\n" + "=" * 70)
print("Checking SDK Default URLs")
print("=" * 70)

# Check what URLs the SDK will use
try:
    from project_x_py.connection_management import ProjectX
    
    print(f"\nChecking ProjectX class defaults...")
    
    # Try to inspect the class
    import inspect
    sig = inspect.signature(ProjectX.__init__)
    print(f"\nProjectX.__init__ parameters:")
    for param_name, param in sig.parameters.items():
        if param.default != inspect.Parameter.empty:
            print(f"  {param_name} = {param.default}")
    
except Exception as e:
    print(f"Could not inspect SDK: {e}")

print("\n" + "=" * 70)
print("Testing Direct API Call to /Auth/loginKey")
print("=" * 70)

import requests
import json

api_url = os.getenv('PROJECTX_API_URL', 'https://api.topstepx.com/api')
auth_url = f"{api_url}/Auth/loginKey"

print(f"\nAuthentication URL: {auth_url}")

payload = {
    "userName": os.getenv('PROJECT_X_USERNAME'),
    "apiKey": os.getenv('PROJECT_X_API_KEY')
}

print(f"Request payload:")
print(f"  userName: {payload['userName']}")
print(f"  apiKey: {payload['apiKey'][:15]}...")

try:
    response = requests.post(auth_url, json=payload, timeout=10)
    print(f"\nResponse Status: {response.status_code}")
    print(f"Response Body: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            print("\n✅ Direct API call succeeded!")
        else:
            print(f"\n❌ API returned error code: {data.get('errorCode')}")
    else:
        print(f"\n❌ HTTP error: {response.status_code}")
        
except Exception as e:
    print(f"\n❌ Request failed: {e}")
