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

# Get contract details
response = requests.post(f'{base_url}/api/Contract/available', headers=headers, json={'live': False}, timeout=30)
data = response.json()

contracts = data.get('contracts', [])
print('🔍 All available contracts:')

# Group by symbol patterns
patterns = {}
for contract in contracts:
    if isinstance(contract, dict):
        symbol_id = contract.get('symbolId', '')
        name = contract.get('name', '')
        contract_id = contract.get('id', '')
        description = contract.get('description', '')
        
        # Extract base symbol pattern
        base = symbol_id.split('.')[-1] if '.' in symbol_id else symbol_id
        if base not in patterns:
            patterns[base] = []
        patterns[base].append({
            'id': contract_id,
            'name': name,
            'description': description,
            'symbolId': symbol_id
        })

# Show all patterns
for pattern, contracts_list in sorted(patterns.items()):
    print(f'\n📊 {pattern} contracts ({len(contracts_list)}):')
    for contract in contracts_list[:3]:  # Show first 3
        print(f'   {contract["id"]} - {contract["name"]} - {contract["description"]}')
    if len(contracts_list) > 3:
        print(f'   ... and {len(contracts_list) - 3} more')

# Look specifically for E-mini futures
print('\n🎯 Looking for E-mini futures (ES, NQ, MES, MNQ):')
emini_found = []
for contract in contracts:
    if isinstance(contract, dict):
        name = contract.get('name', '').upper()
        desc = contract.get('description', '').upper()
        symbol_id = contract.get('symbolId', '').upper()
        
        search_terms = ['ES', 'SP', 'S&P', 'NASDAQ', 'NQ', 'MES', 'MNQ', 'EMINI', 'E-MINI', 'EP', 'ENQ']
        if any(term in name + ' ' + desc + ' ' + symbol_id for term in search_terms):
            emini_found.append(contract)
            print(f'   {contract["id"]} - {contract["name"]} - {contract["description"]}')

if not emini_found:
    print('   ❌ No E-mini contracts found. Let\'s check all contract names:')
    for contract in contracts[:10]:  # Show first 10
        print(f'   {contract["id"]} - {contract["name"]} - {contract["description"]}')

# Now test Contract/search to find ES specifically
print('\n🔎 Testing Contract/search for ES:')
search_response = requests.post(f'{base_url}/api/Contract/search', 
                              headers=headers, 
                              json={'searchText': 'ES', 'live': False}, 
                              timeout=30)

if search_response.status_code == 200:
    search_data = search_response.json()
    search_contracts = search_data.get('contracts', [])
    print(f'Found {len(search_contracts)} contracts with "ES":')
    for contract in search_contracts:
        print(f'   {contract["id"]} - {contract["name"]} - {contract["description"]}')
else:
    print(f'Search failed: {search_response.status_code} - {search_response.text}')

print('\n🔎 Testing Contract/search for NQ:')
search_response2 = requests.post(f'{base_url}/api/Contract/search', 
                               headers=headers, 
                               json={'searchText': 'NQ', 'live': False}, 
                               timeout=30)

if search_response2.status_code == 200:
    search_data2 = search_response2.json()
    search_contracts2 = search_data2.get('contracts', [])
    print(f'Found {len(search_contracts2)} contracts with "NQ":')
    for contract in search_contracts2:
        print(f'   {contract["id"]} - {contract["name"]} - {contract["description"]}')
else:
    print(f'Search failed: {search_response2.status_code} - {search_response2.text}')