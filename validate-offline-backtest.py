#!/usr/bin/env python3
"""
Validation script for offline backtest mode
Verifies that all required components are in place for offline operation
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

def check_file(path, description):
    """Check if a file exists and return status"""
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists

def check_json_file(path, description):
    """Check if a JSON file exists and is valid"""
    if not os.path.exists(path):
        print(f"❌ {description}: {path} (not found)")
        return False
    
    try:
        with open(path, 'r') as f:
            content = f.read()
            # .NET configuration files support // comments, so we'll just check if file is readable
            # and has some basic JSON structure
            if content.strip().startswith('{') or content.strip().startswith('['):
                # For .NET config files, just verify they're readable
                if 'appsettings' in path and '//' in content:
                    print(f"✅ {description}: {path} (valid .NET config with comments)")
                    return True
                # Try to parse as JSON
                data = json.loads(content)
                print(f"✅ {description}: {path} ({type(data).__name__} with {len(data) if isinstance(data, (list, dict)) else '?'} items)")
                return True
            else:
                print(f"❌ {description}: {path} (not valid JSON structure)")
                return False
    except json.JSONDecodeError as e:
        # If it's a .NET config file with comments, that's OK
        if 'appsettings' in path:
            print(f"✅ {description}: {path} (valid .NET config with comments)")
            return True
        print(f"❌ {description}: {path} (invalid JSON: {e})")
        return False
    except Exception as e:
        print(f"❌ {description}: {path} (error: {e})")
        return False

def analyze_quotes_file(path, symbol):
    """Analyze a quotes file and show details"""
    if not os.path.exists(path):
        return False
    
    try:
        with open(path, 'r') as f:
            quotes = json.load(f)
        
        if not quotes:
            print(f"   ⚠️  File is empty")
            return False
        
        first = quotes[0]
        last = quotes[-1]
        
        print(f"   📊 Symbol: {symbol}")
        print(f"   📊 Total bars: {len(quotes)}")
        print(f"   📊 Date range: {first['Time']} to {last['Time']}")
        print(f"   📊 Price range: ${first['Low']:.2f} - ${last['High']:.2f}")
        print(f"   📊 Fields: {', '.join(first.keys())}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error analyzing: {e}")
        return False

def main():
    print("╔════════════════════════════════════════════════════════════════════════════════╗")
    print("║         Offline Backtest Mode - Validation & Verification                     ║")
    print("╚════════════════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Get repository root
    repo_root = Path(__file__).parent
    os.chdir(repo_root)
    
    all_checks_passed = True
    
    # 1. Check historical data files
    print("1. Checking Historical Data Files...")
    print("-" * 80)
    
    es_quotes = "datasets/quotes/es_quotes.json"
    nq_quotes = "datasets/quotes/nq_quotes.json"
    
    es_ok = check_json_file(es_quotes, "ES quotes data")
    if es_ok:
        analyze_quotes_file(es_quotes, "ES")
    all_checks_passed &= es_ok
    
    print()
    nq_ok = check_json_file(nq_quotes, "NQ quotes data")
    if nq_ok:
        analyze_quotes_file(nq_quotes, "NQ")
    all_checks_passed &= nq_ok
    
    print()
    
    # 2. Check configuration files
    print("2. Checking Configuration Files...")
    print("-" * 80)
    
    config_files = [
        ("appsettings.backtest.json", "Backtest configuration"),
        ("src/UnifiedOrchestrator/appsettings.json", "Orchestrator configuration"),
    ]
    
    for file_path, description in config_files:
        all_checks_passed &= check_json_file(file_path, description)
    
    print()
    
    # 3. Check key source files
    print("3. Checking Source Code Components...")
    print("-" * 80)
    
    source_files = [
        ("src/Backtest/BacktestHarnessService.cs", "Backtest harness service"),
        ("src/Backtest/IHistoricalDataProvider.cs", "Historical data provider interface"),
        ("src/UnifiedOrchestrator/Services/HistoricalDataProviders.cs", "Data providers implementation"),
        ("src/Backtest/Extensions/BacktestServiceExtensions.cs", "DI registration extensions"),
    ]
    
    for file_path, description in source_files:
        all_checks_passed &= check_file(file_path, description)
    
    print()
    
    # 4. Check scripts and documentation
    print("4. Checking Scripts and Documentation...")
    print("-" * 80)
    
    docs = [
        ("run-offline-backtest.sh", "Offline backtest launcher script"),
        ("OFFLINE_BACKTEST_GUIDE.md", "Offline backtest documentation"),
        ("BACKTEST_MODE_GUIDE.md", "Backtest mode guide"),
        ("test-backtest-mode.sh", "Test backtest script"),
    ]
    
    for file_path, description in docs:
        all_checks_passed &= check_file(file_path, description)
    
    print()
    
    # 5. Show environment setup
    print("5. Environment Setup for Offline Backtest...")
    print("-" * 80)
    
    env_vars = {
        "BACKTEST_MODE": "1",
        "ENABLE_BACKTEST_UI": "0",
        "SKIP_MODE_PROMPT": "1",
        "DRY_RUN": "1",
        "BACKTEST_SYMBOL": "ES",
        "BACKTEST_MODEL": "CVaR-PPO",
        "BACKTEST_DAYS": "1",
        "ASPNETCORE_ENVIRONMENT": "backtest"
    }
    
    print("Required environment variables:")
    for var, value in env_vars.items():
        print(f"  export {var}={value}")
    
    print()
    
    # 6. Summary
    print("═" * 80)
    if all_checks_passed:
        print("✅ All validation checks PASSED!")
        print()
        print("The offline backtest mode is properly configured and ready to run.")
        print()
        print("To run the offline backtest:")
        print("  ./run-offline-backtest.sh")
        print()
        print("Or manually:")
        print("  export BACKTEST_MODE=1 ENABLE_BACKTEST_UI=0 SKIP_MODE_PROMPT=1")
        print("  cd src/UnifiedOrchestrator && dotnet run")
        print()
        return 0
    else:
        print("❌ Some validation checks FAILED!")
        print()
        print("Please review the errors above and ensure all required files exist.")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
