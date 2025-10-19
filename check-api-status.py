"""
Check TopstepX API status and maintenance
"""
import requests
import json
from datetime import datetime

print("=" * 70)
print("TopstepX API Status Check")
print("=" * 70)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Test various endpoints to see if API is up
endpoints_to_test = [
    ("Main API", "https://api.topstepx.com"),
    ("API Base", "https://api.topstepx.com/api"),
    ("RTC Hub", "https://rtc.topstepx.com"),
    ("Website", "https://topstepx.com"),
]

print("Testing TopstepX Service Availability:")
print("-" * 70)

all_down = True
any_up = False

for name, url in endpoints_to_test:
    try:
        response = requests.get(url, timeout=10, allow_redirects=True)
        status = response.status_code
        
        if status == 200:
            print(f"✅ {name:20} UP (HTTP {status})")
            any_up = True
            all_down = False
        elif status == 401 or status == 403:
            print(f"✅ {name:20} UP (HTTP {status} - Auth required, but server responding)")
            any_up = True
            all_down = False
        elif status == 503:
            print(f"🔧 {name:20} MAINTENANCE (HTTP {status})")
        elif status == 502 or status == 504:
            print(f"❌ {name:20} DOWN (HTTP {status} - Gateway error)")
        else:
            print(f"⚠️  {name:20} UNKNOWN (HTTP {status})")
            any_up = True
            all_down = False
            
    except requests.exceptions.Timeout:
        print(f"⏱️  {name:20} TIMEOUT (Server not responding)")
    except requests.exceptions.ConnectionError as e:
        print(f"❌ {name:20} DOWN (Connection refused)")
    except Exception as e:
        print(f"❌ {name:20} ERROR ({type(e).__name__})")

print()
print("=" * 70)

if all_down:
    print("🔧 TopstepX API appears to be DOWN or under MAINTENANCE")
    print("   → This would explain authentication failures")
    print("   → Wait and try again later")
    print("   → Check TopstepX status page or Twitter/social media")
elif any_up:
    print("✅ TopstepX API is UP and responding")
    print("   → Authentication issue is likely API key related")
    print("   → Error code 3 = Invalid/Unauthorized API key")
    print("   → Check API key status in TopstepX portal")

# Test the specific auth endpoint
print()
print("=" * 70)
print("Testing Authentication Endpoint Specifically:")
print("-" * 70)

auth_url = "https://api.topstepx.com/api/Auth/loginKey"
print(f"URL: {auth_url}")

try:
    # Send empty auth request to see if endpoint is responding
    response = requests.post(
        auth_url,
        json={"userName": "", "apiKey": ""},
        headers={"Content-Type": "application/json"},
        timeout=10
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        
        if data.get('errorCode') == 3:
            print("\n✅ Auth endpoint is WORKING (returned error code 3 for empty creds)")
            print("   → This confirms the API is UP")
            print("   → Your API key is being rejected as invalid")
        else:
            print(f"\n⚠️  Unexpected response: {data}")
    elif response.status_code == 503:
        print("\n🔧 Auth endpoint returning 503 - MAINTENANCE MODE")
    else:
        print(f"\n⚠️  Unexpected status: {response.status_code}")
        print(f"Response: {response.text[:200]}")
        
except requests.exceptions.Timeout:
    print("⏱️  TIMEOUT - Auth endpoint not responding (possible maintenance)")
except requests.exceptions.ConnectionError:
    print("❌ CONNECTION REFUSED - Auth endpoint down (possible maintenance)")
except Exception as e:
    print(f"❌ ERROR: {e}")

print()
print("=" * 70)
