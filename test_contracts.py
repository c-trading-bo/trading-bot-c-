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

# Try Contract/available endpoint with POST
print('Testing Contract/available endpoint...')
payload = {"live": False}
response = requests.post(f'{base_url}/api/Contract/available', headers=headers, json=payload, timeout=30)
print(f'Response: {response.status_code} {response.reason}')

if response.status_code == 200:
    data = response.json()
    print(f'Success! Found {len(data.get("data", []))} contracts')
    print(f'Full response: {data}')
    
    # Look for ES and NQ contracts
    if 'data' in data and isinstance(data['data'], list):
        es_contracts = []
        nq_contracts = []
        
        for contract in data['data']:
            if isinstance(contract, dict) and 'symbol' in contract:
                symbol = contract.get('symbol', '')
                contract_id = contract.get('id', 'N/A')
                
                if 'ES' in symbol:
                    es_contracts.append((symbol, contract_id))
                elif 'NQ' in symbol or 'MNQ' in symbol:
                    nq_contracts.append((symbol, contract_id))
        
        print(f'\nES contracts found: {len(es_contracts)}')
        for symbol, contract_id in es_contracts:
            print(f'  {symbol}: {contract_id}')
            
        print(f'\nNQ/MNQ contracts found: {len(nq_contracts)}')
        for symbol, contract_id in nq_contracts:
            print(f'  {symbol}: {contract_id}')
            
        # Find MES and MNQ specifically
        mes_id = None
        mnq_id = None
        
        for symbol, contract_id in es_contracts:
            if symbol == 'MES':
                mes_id = contract_id
                break
                
        for symbol, contract_id in nq_contracts:
            if symbol == 'MNQ':
                mnq_id = contract_id
                break
                
        if mes_id:
            print(f'\n✅ MES contract ID: {mes_id}')
        if mnq_id:
            print(f'✅ MNQ contract ID: {mnq_id}')
            
# Try with live=true as well
print('\n\nTesting with live=true...')
payload_live = {"live": True}
response_live = requests.post(f'{base_url}/api/Contract/available', headers=headers, json=payload_live, timeout=30)
print(f'Live response: {response_live.status_code} {response_live.reason}')

if response_live.status_code == 200:
    data_live = response_live.json()
    print(f'Live contracts found: {len(data_live.get("data", []))}')
    if 'data' in data_live and len(data_live['data']) > 0:
        print(f'First few live contracts:')
        for contract in data_live['data'][:5]:
            if isinstance(contract, dict):
                symbol = contract.get('symbol', 'N/A')
                contract_id = contract.get('id', 'N/A')
                print(f'  {symbol}: {contract_id}')
else:
    print(f'Error: {response.text}')