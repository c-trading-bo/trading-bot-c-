"""
Raw API test - Shows exact response from TopstepX authentication endpoint
"""
import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

api_key = os.getenv('PROJECT_X_API_KEY')
username = os.getenv('PROJECT_X_USERNAME')

print("=" * 70)
print("TopstepX Raw API Authentication Test")
print("=" * 70)
print(f"\nUsername: {username}")
print(f"API Key: {api_key[:10]}...{api_key[-10:]}")
print(f"\nEndpoint: https://api.topstepx.com/api/Auth/loginKey")

payload = {
    "userName": username,
    "apiKey": api_key
}

print(f"\nRequest body:")
print(json.dumps(payload, indent=2))

print("\n" + "-" * 70)
print("Sending authentication request...")
print("-" * 70)

try:
    response = requests.post(
        "https://api.topstepx.com/api/Auth/loginKey",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=10
    )
    
    print(f"\nHTTP Status: {response.status_code}")
    print(f"Response Headers:")
    for key, value in response.headers.items():
        if key.lower() not in ['set-cookie', 'cookie']:
            print(f"  {key}: {value}")
    
    print(f"\nResponse Body:")
    response_data = response.json()
    print(json.dumps(response_data, indent=2))
    
    print("\n" + "=" * 70)
    print("ANALYSIS:")
    print("=" * 70)
    
    if response_data.get("success") == True and response_data.get("token"):
        print("✅ SUCCESS: Authentication working!")
        print(f"   Token received: {response_data['token'][:20]}...")
    elif response_data.get("errorCode") == 3:
        print("❌ ERROR CODE 3: Invalid or Unauthorized API Key")
        print("\nPossible reasons:")
        print("  1. API key not activated in TopstepX portal")
        print("  2. API key lacks required permissions (Trading/Market Data)")
        print("  3. Account not enabled for API access")
        print("  4. API key expired or revoked")
        print("\n✅ SOLUTION:")
        print("  → Log into https://topstepx.com")
        print("  → Go to Account Settings → API Keys")
        print("  → Verify key status shows 'Active'")
        print("  → Check permissions are enabled")
        print("  → If inactive, regenerate a new key")
    else:
        print(f"⚠️  Unexpected response: {response_data}")
    
except Exception as e:
    print(f"\n❌ Request failed: {e}")
    print(f"   Error type: {type(e).__name__}")

print("\n" + "=" * 70)
