#!/usr/bin/env python3
"""Simple API fallback handler"""
import requests
import json
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
