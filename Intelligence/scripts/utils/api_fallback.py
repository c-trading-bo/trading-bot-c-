#!/usr/bin/env python3
"""Simple API fallback handler"""
import requests
import json
import sys
from datetime import datetime

def fetch_with_fallback(url, params=None, timeout=10):
    """Fetch data with simple fallback to mock data"""
    try:
        response = requests.get(url, params=params, timeout=timeout)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    # Return mock data on failure
    return {
        'status': 'mock',
        'timestamp': datetime.utcnow().isoformat(),
        'data': 'Mock data - API unavailable'
    }

def get_mock_news():
    """Get mock news data"""
    return {
        'status': 'mock',
        'timestamp': datetime.utcnow().isoformat(),
        'articles': [
            {
                'title': 'Market Update - Mock Data',
                'description': 'Mock news article for testing purposes',
                'publishedAt': datetime.utcnow().isoformat(),
                'source': {'name': 'Mock News'},
                'sentiment': 0.0
            }
        ],
        'totalResults': 1
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_type = sys.argv[1]
        if api_type == 'news':
            result = get_mock_news()
        else:
            result = fetch_with_fallback('https://httpbin.org/status/404')  # This will fail and return mock
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python api_fallback.py [news|market|economic]")
        print("Testing news API fallback...")
        result = get_mock_news()
        print(json.dumps(result, indent=2))
