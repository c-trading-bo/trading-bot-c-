"""
Test which URLs the SDK is actually calling
"""
import asyncio
import os
from dotenv import load_dotenv
import aiohttp
from unittest.mock import patch, AsyncMock
import json

load_dotenv()

api_key = os.getenv('PROJECT_X_API_KEY')
username = os.getenv('PROJECT_X_USERNAME')

print("=" * 70)
print("Tracking TopstepX SDK API Calls")
print("=" * 70)
print(f"API Key: {api_key[:15]}...")
print(f"Username: {username}")
print()

# Track all HTTP requests
requests_made = []

# Patch aiohttp to intercept requests
original_request = aiohttp.ClientSession.request

async def tracked_request(self, method, url, **kwargs):
    print(f"\n📡 SDK REQUEST:")
    print(f"   Method: {method}")
    print(f"   URL: {url}")
    
    if 'json' in kwargs:
        print(f"   Body: {json.dumps(kwargs['json'], indent=6)}")
    
    if 'headers' in kwargs:
        headers = kwargs['headers'].copy()
        # Mask sensitive headers
        if 'Authorization' in headers:
            headers['Authorization'] = headers['Authorization'][:20] + '...'
        print(f"   Headers: {json.dumps(dict(headers), indent=6)}")
    
    requests_made.append({
        'method': method,
        'url': url,
        'body': kwargs.get('json'),
        'headers': {k: v for k, v in kwargs.get('headers', {}).items()}
    })
    
    # Call the original request
    try:
        response = await original_request(self, method, url, **kwargs)
        print(f"   Response: {response.status}")
        
        # Try to read response body
        try:
            body = await response.text()
            response_json = json.loads(body)
            print(f"   Response Body: {json.dumps(response_json, indent=6)}")
        except:
            pass
        
        return response
    except Exception as e:
        print(f"   Error: {e}")
        raise

async def test_with_tracking():
    # Patch the request method
    with patch.object(aiohttp.ClientSession, 'request', tracked_request):
        print("\n" + "=" * 70)
        print("Attempting SDK Authentication...")
        print("=" * 70)
        
        try:
            from project_x_py import TradingSuite
            suite = await TradingSuite.from_env(['ES'])
            print("\n✅ Authentication succeeded!")
        except Exception as e:
            print(f"\n❌ Authentication failed: {e}")
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: {len(requests_made)} requests made")
    print("=" * 70)
    
    for i, req in enumerate(requests_made, 1):
        print(f"\n{i}. {req['method']} {req['url']}")

if __name__ == "__main__":
    asyncio.run(test_with_tracking())
