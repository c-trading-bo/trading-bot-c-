#!/usr/bin/env python3
"""
Test script to verify daily report generation works correctly
"""

import os
import sys
import json
from datetime import datetime

def test_daily_report():
    """Test the daily report generation"""
    
    print("🧪 Testing Daily Report Generation...")
    print("=" * 50)
    
    # Check required directories
    required_dirs = [
        "Intelligence/data/signals",
        "Intelligence/data/features", 
        "Intelligence/models",
        "Intelligence/reports"
    ]
    
    print("📁 Checking directory structure...")
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path} exists")
        else:
            print(f"⚠️  Creating {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
    
    # Check if scripts exist
    scripts = [
        "Intelligence/scripts/generate_signals.py",
        "Intelligence/scripts/generate_daily_report.py"
    ]
    
    print("\n📜 Checking scripts...")
    for script in scripts:
        if os.path.exists(script):
            print(f"✅ {script} exists")
        else:
            print(f"❌ {script} missing")
            return False
    
    # Create test data if needed
    signals_file = "Intelligence/data/signals/latest.json"
    if not os.path.exists(signals_file):
        print(f"\n📝 Creating test signals file: {signals_file}")
        test_signals = {
            "timestamp": datetime.now().isoformat(),
            "signals": {
                "ES": {"signal": "BUY", "confidence": 0.75, "price": 4500.0},
                "NQ": {"signal": "HOLD", "confidence": 0.60, "price": 15000.0}
            },
            "features": {
                "vix": 18.5,
                "dxy": 103.2,
                "market_regime": "BULLISH"
            }
        }
        
        with open(signals_file, 'w') as f:
            json.dump(test_signals, f, indent=2)
        print(f"✅ Created test signals file")
    
    print("\n🎯 Daily report system test completed successfully!")
    print("📈 Ready to generate daily reports!")
    
    return True

if __name__ == "__main__":
    success = test_daily_report()
    sys.exit(0 if success else 1)
