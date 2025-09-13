#!/usr/bin/env python3
"""
TopstepX Contract Discovery Tool
This script uses the same authentication method as the C# app to discover contracts
"""

import requests
import json
import os
import sys
from typing import Dict, Any, Optional

class TopstepXContractDiscovery:
    def __init__(self):
        self.base_url = "https://api.topstepx.com"
        self.jwt_token = None
        self.session = requests.Session()
        
    def authenticate(self) -> bool:
        """Authenticate with TopstepX using environment credentials"""
        username = os.environ.get('TOPSTEPX_USERNAME')
        api_key = os.environ.get('TOPSTEPX_API_KEY')
        
        if not username or not api_key:
            print("❌ Missing credentials: Set TOPSTEPX_USERNAME and TOPSTEPX_API_KEY environment variables")
            return False
            
        print(f"🔐 Authenticating with username: {username}")
        
        auth_data = {
            "username": username,
            "apiKey": api_key
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/Auth/login",
                json=auth_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.jwt_token = data.get('token')
                if self.jwt_token:
                    print(f"✅ Authentication successful! JWT token length: {len(self.jwt_token)}")
                    self.session.headers.update({"Authorization": f"Bearer {self.jwt_token}"})
                    return True
                else:
                    print("❌ No token in auth response")
                    return False
            else:
                print(f"❌ Authentication failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return False
    
    def test_endpoint(self, method: str, path: str, body: Optional[Dict[str, Any]] = None, name: str = "") -> bool:
        """Test a specific API endpoint"""
        print(f"📡 Testing: {name}")
        print(f"   {method} {path}")
        
        try:
            url = f"{self.base_url}{path}"
            
            if method.upper() == "GET":
                response = self.session.get(url, timeout=10)
            else:
                response = self.session.post(url, json=body, timeout=10)
                
            if response.status_code == 200:
                print("   ✅ SUCCESS")
                
                try:
                    data = response.json()
                    self.parse_contract_data(data)
                    return True
                except json.JSONDecodeError:
                    print("   📄 Response not JSON")
                    return True
                    
            else:
                print(f"   ❌ FAILED: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            return False
    
    def parse_contract_data(self, data: Any) -> None:
        """Parse response data looking for contract information"""
        contracts_found = []
        
        def extract_contracts(obj, path=""):
            if isinstance(obj, dict):
                # Look for contract ID fields
                contract_id = None
                symbol = None
                
                for id_field in ['contractId', 'id', 'instrumentId', 'contract_id']:
                    if id_field in obj:
                        contract_id = obj[id_field]
                        break
                        
                for symbol_field in ['symbol', 'instrument', 'name', 'ticker']:
                    if symbol_field in obj:
                        symbol = obj[symbol_field]
                        break
                
                if contract_id and symbol:
                    contracts_found.append((symbol, contract_id))
                
                # Recursively check nested objects
                for key, value in obj.items():
                    extract_contracts(value, f"{path}.{key}" if path else key)
                    
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_contracts(item, f"{path}[{i}]" if path else f"[{i}]")
        
        extract_contracts(data)
        
        if contracts_found:
            for symbol, contract_id in contracts_found:
                print(f"   📋 Contract: {symbol} -> ID: {contract_id}")
        else:
            print("   📄 No contract data found")
    
    def discover_contracts(self) -> None:
        """Run contract discovery across multiple endpoints"""
        print("🔍 TopstepX Contract Discovery Tool")
        print(f"Base URL: {self.base_url}")
        print()
        
        if not self.authenticate():
            return
            
        print()
        print("🚀 Starting endpoint discovery...")
        print()
        
        # Define endpoints to test
        endpoints = [
            ("POST", "/api/Contract/available", {"live": False}, "Contract Available (Sim)"),
            ("POST", "/api/Contract/available", {"live": True}, "Contract Available (Live)"),
            ("POST", "/api/Contract/search", {"symbol": "ES"}, "Contract Search (ES)"),
            ("POST", "/api/Contract/search", {"symbol": "NQ"}, "Contract Search (NQ)"),
            ("POST", "/api/Contract/search", {"search": "ES"}, "Contract Search (ES - search field)"),
            ("POST", "/api/Contract/search", {"query": "ES"}, "Contract Search (ES - query field)"),
            ("POST", "/api/Instrument/search", {"symbol": "ES"}, "Instrument Search (ES)"),
            ("POST", "/api/Instrument/search", {"symbol": "NQ"}, "Instrument Search (NQ)"),
            ("POST", "/api/Instrument/available", {}, "Instrument Available"),
            ("GET", "/api/Instrument/ES", None, "Instrument Direct (ES)"),
            ("GET", "/api/Instrument/NQ", None, "Instrument Direct (NQ)"),
            ("GET", "/api/Quote/ES", None, "Quote Direct (ES)"),
            ("GET", "/api/Quote/NQ", None, "Quote Direct (NQ)"),
            ("GET", "/api/Quotes/ES", None, "Quotes Direct (ES)"),
            ("GET", "/api/MarketData/ES", None, "MarketData Direct (ES)"),
            ("POST", "/api/Account/instruments", {}, "Account Instruments"),
            ("POST", "/api/Account/contracts", {}, "Account Contracts"),
            ("POST", "/api/Position/search", {}, "Position Search"),
            ("GET", "/api/Positions", None, "Positions List"),
            ("POST", "/api/Order/search", {}, "Order Search"),
            ("GET", "/api/contracts", None, "Generic Contracts"),
            ("GET", "/api/instruments", None, "Generic Instruments"),
            ("GET", "/api/symbols", None, "Generic Symbols"),
        ]
        
        success_count = 0
        for method, path, body, name in endpoints:
            if self.test_endpoint(method, path, body, name):
                success_count += 1
            print()
        
        print("🎯 Discovery complete!")
        print(f"📊 {success_count}/{len(endpoints)} endpoints responded successfully")
        print()
        print("💡 If no contracts were found:")
        print("1. Market might be closed (normal for eval accounts)")
        print("2. Try during market hours when TopstepX streams are active")
        print("3. Use TopstepX web app DevTools to capture contract IDs")
        print("4. Set environment variables with real contract IDs:")
        print("   $env:TOPSTEPX_EVAL_ES_ID='12345678'")
        print("   $env:TOPSTEPX_EVAL_NQ_ID='23456789'")

if __name__ == "__main__":
    discovery = TopstepXContractDiscovery()
    discovery.discover_contracts()