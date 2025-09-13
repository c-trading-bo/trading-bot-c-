#!/usr/bin/env python3

from dotenv import load_dotenv
import os
import requests
import json

load_dotenv()

jwt_token = os.getenv('TOPSTEPX_JWT')
base_url = os.getenv('TOPSTEPX_API_BASE', 'https://api.topstepx.com')

headers = {
    'Authorization': f'Bearer {jwt_token}',
    'Content-Type': 'application/json'
}

# Try Contract/search endpoint
print('Testing Contract/search endpoint...')
payload = {
    "request": {
        "searchText": "ES",
        "live": False,
        "symbols": ["ES", "MES", "NQ", "MNQ"],
        "includeExpired": False
    }
}
response = requests.post(f'{base_url}/api/Contract/search', headers=headers, json=payload, timeout=30)
print(f'Response: {response.status_code} {response.reason}')

if response.status_code == 200:
    data = response.json()
    print(f'Success! Response structure:')
    print(f'Keys: {list(data.keys()) if isinstance(data, dict) else "Not a dict"}')
    
    if 'data' in data and isinstance(data['data'], list):
        print(f'Found {len(data["data"])} contracts')
        
        for contract in data['data']:
            if isinstance(contract, dict):
                symbol = contract.get('symbol', 'N/A')
                contract_id = contract.get('id', 'N/A')
                print(f'  {symbol}: {contract_id}')
    else:
        print(f'Full response: {json.dumps(data, indent=2)[:1000]}...')
else:
    print(f'Error: {response.text}')

# Also try different search patterns
print('\nTrying broader search...')
payload2 = {
    "request": {
        "searchText": "",
        "live": False,
        "symbols": [],
        "includeExpired": False
    }
}
response2 = requests.post(f'{base_url}/api/Contract/search', headers=headers, json=payload2, timeout=30)
print(f'Broader search response: {response2.status_code} {response2.reason}')

if response2.status_code == 200:
    data2 = response2.json()
    if 'data' in data2 and isinstance(data2['data'], list):
        print(f'Broader search found {len(data2["data"])} contracts')
        # Show first few contracts
        for i, contract in enumerate(data2['data'][:10]):
            if isinstance(contract, dict):
                symbol = contract.get('symbol', 'N/A')
                contract_id = contract.get('id', 'N/A')
                print(f'  {symbol}: {contract_id}')
        if len(data2['data']) > 10:
            print(f'  ... and {len(data2["data"]) - 10} more')