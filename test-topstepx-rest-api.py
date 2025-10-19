#!/usr/bin/env python3
"""Test TopstepX REST API endpoints to find working alternatives to WebSocket"""

import os
import sys
import requests
from typing import Dict, List, Tuple

def test_rest_endpoints() -> None:
    """Test various REST API endpoint patterns"""
    
    api_key = os.getenv('TOPSTEPX_API_KEY')
    username = os.getenv('TOPSTEPX_USERNAME')
    account_id = os.getenv('TOPSTEPX_ACCOUNT_ID')
    
    print("=" * 70)
    print("TopstepX REST API Endpoint Discovery")
    print("=" * 70)
    print(f"API Key: {api_key[:20]}..." if api_key else "API Key: NOT SET")
    print(f"Username: {username}")
    print(f"Account ID: {account_id}")
    print()
    
    # Base URLs to test
    base_urls = [
        'https://api.topstepx.com',
        'https://rtc.topstepx.com',
    ]
    
    # API endpoint patterns to test
    endpoints = [
        # Account endpoints
        '/api/v1/accounts',
        f'/api/v1/accounts/{account_id}',
        '/api/accounts',
        f'/api/accounts/{account_id}',
        '/v1/accounts',
        f'/v1/accounts/{account_id}',
        '/accounts',
        f'/accounts/{account_id}',
        
        # Position endpoints
        '/api/v1/positions',
        f'/api/v1/accounts/{account_id}/positions',
        '/api/positions',
        '/v1/positions',
        '/positions',
        
        # Order endpoints
        '/api/v1/orders',
        f'/api/v1/accounts/{account_id}/orders',
        '/api/orders',
        '/v1/orders',
        '/orders',
        
        # Market data endpoints
        '/api/v1/market/quotes',
        '/api/v1/market/data',
        '/api/market/quotes',
        '/api/market/data',
        '/market/quotes',
        '/market/data',
        
        # User/Auth endpoints
        '/api/v1/user',
        '/api/v1/user/profile',
        '/api/user',
        '/user',
        '/me',
    ]
    
    # Different header patterns to test
    auth_patterns = [
        {'Authorization': f'Bearer {api_key}'},
        {'X-API-Key': api_key},
        {'Api-Key': api_key},
        {'apikey': api_key},
        {'Authorization': f'ApiKey {api_key}'},
        {'X-Username': username, 'X-API-Key': api_key},
    ]
    
    results: Dict[str, List[Tuple[str, int, str]]] = {
        '2xx Success': [],
        '3xx Redirect': [],
        '4xx Client Error': [],
        '5xx Server Error': [],
        'Other': []
    }
    
    total_tests = len(base_urls) * len(endpoints) * len(auth_patterns)
    current_test = 0
    
    print(f"Testing {total_tests} endpoint combinations...")
    print()
    
    for base_url in base_urls:
        for endpoint in endpoints:
            for auth_headers in auth_patterns:
                current_test += 1
                url = f"{base_url}{endpoint}"
                headers = {**auth_headers, 'Content-Type': 'application/json'}
                
                try:
                    response = requests.get(url, headers=headers, timeout=5)
                    status = response.status_code
                    auth_type = list(auth_headers.keys())[0]
                    
                    # Categorize by status code
                    if 200 <= status < 300:
                        results['2xx Success'].append((url, status, auth_type))
                        print(f"✅ {status} {url} (auth: {auth_type})")
                    elif 300 <= status < 400:
                        results['3xx Redirect'].append((url, status, auth_type))
                    elif 400 <= status < 500:
                        results['4xx Client Error'].append((url, status, auth_type))
                    elif 500 <= status < 600:
                        results['5xx Server Error'].append((url, status, auth_type))
                    else:
                        results['Other'].append((url, status, auth_type))
                        
                except requests.exceptions.RequestException as e:
                    # Connection errors, timeouts, etc.
                    pass
                
                # Progress indicator
                if current_test % 50 == 0:
                    print(f"Progress: {current_test}/{total_tests} tests completed...")
    
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    # Print successful endpoints first
    if results['2xx Success']:
        print()
        print("✅ WORKING ENDPOINTS (2xx Success):")
        print("-" * 70)
        for url, status, auth_type in results['2xx Success']:
            print(f"  {status} {url}")
            print(f"      Auth: {auth_type}")
    else:
        print()
        print("❌ No working endpoints found (no 2xx responses)")
    
    # Print other interesting results
    if results['3xx Redirect']:
        print()
        print("↪️  REDIRECTS (3xx):")
        print("-" * 70)
        for url, status, auth_type in results['3xx Redirect'][:10]:  # Limit to 10
            print(f"  {status} {url} (auth: {auth_type})")
    
    # Show some 4xx errors (might indicate endpoint exists but wrong auth)
    if results['4xx Client Error']:
        print()
        print("⚠️  CLIENT ERRORS (4xx) - Top 10:")
        print("-" * 70)
        # Group by status code
        status_groups: Dict[int, List[Tuple[str, str]]] = {}
        for url, status, auth_type in results['4xx Client Error']:
            if status not in status_groups:
                status_groups[status] = []
            status_groups[status].append((url, auth_type))
        
        for status in sorted(status_groups.keys()):
            print(f"  Status {status}: {len(status_groups[status])} endpoints")
            for url, auth_type in status_groups[status][:3]:  # Show first 3
                print(f"    {url} (auth: {auth_type})")
    
    if results['5xx Server Error']:
        print()
        print("🔥 SERVER ERRORS (5xx):")
        print("-" * 70)
        for url, status, auth_type in results['5xx Server Error'][:10]:
            print(f"  {status} {url} (auth: {auth_type})")
    
    print()
    print("=" * 70)
    print(f"Total tests: {total_tests}")
    print(f"Successful: {len(results['2xx Success'])}")
    print(f"Client errors: {len(results['4xx Client Error'])}")
    print(f"Server errors: {len(results['5xx Server Error'])}")
    print("=" * 70)

if __name__ == '__main__':
    test_rest_endpoints()
