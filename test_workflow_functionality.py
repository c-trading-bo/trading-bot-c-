#!/usr/bin/env python3
"""
Workflow Functionality Test
Tests all enabled workflow scripts to ensure they work properly
"""

import os
import sys
import traceback
import yfinance as yf
import pandas as pd
import numpy as np

def test_basic_imports():
    """Test all required imports"""
    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np
        import scipy
        import ta
        import json
        import logging
        from datetime import datetime, timedelta
        print("✅ All basic imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_data_fetch():
    """Test data fetching"""
    try:
        ticker = yf.Ticker('ES=F')
        data = ticker.history(period='1d', interval='1h')
        if len(data) > 0:
            print(f"✅ Data fetch successful: {len(data)} bars")
            return True
        else:
            print("❌ No data fetched")
            return False
    except Exception as e:
        print(f"❌ Data fetch failed: {e}")
        return False

def test_zones_script():
    """Test zones identifier"""
    try:
        sys.path.append('Intelligence/scripts')
        exec(open('Intelligence/scripts/identify_zones.py').read())
        zi = globals()['SupplyDemandZoneIdentifier']('ES=F', 1)
        print("✅ Zones identifier script works")
        return True
    except Exception as e:
        print(f"❌ Zones script failed: {e}")
        return False

def test_data_directories():
    """Test data directory creation"""
    try:
        dirs = [
            'Intelligence/data/correlations',
            'Intelligence/data/overnight', 
            'Intelligence/data/zones',
            'Intelligence/data/volatility',
            'Intelligence/data/microstructure'
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        print("✅ All data directories created")
        return True
    except Exception as e:
        print(f"❌ Directory creation failed: {e}")
        return False

def main():
    print("🔍 WORKFLOW FUNCTIONALITY TEST")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Data Fetching", test_data_fetch),  
        ("Zones Script", test_zones_script),
        ("Data Directories", test_data_directories)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\nTesting {name}...")
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - WORKFLOWS ARE READY!")
        return True
    else:
        print("❌ SOME TESTS FAILED - CHECK CONFIGURATION")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
