#!/usr/bin/env python3
"""
Quick Start Launcher for Local Bot Mechanic
Fastest way to get your bot running
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Setup the environment for the bot mechanic"""
    print("🔧 Setting up Local Bot Mechanic environment...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        return False
    
    # Create necessary directories
    dirs = [
        "Intelligence/mechanic/database",
        "Intelligence/mechanic/logs",
        "Intelligence/mechanic/reports",
        "Intelligence/scripts/strategies",
        "Intelligence/scripts/ml",
        "Intelligence/scripts/data",
        "Intelligence/data",
        "Intelligence/models"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Install required packages
    required = [
        'pandas', 'numpy', 'scikit-learn', 'requests', 
        'yfinance', 'ta', 'matplotlib', 'seaborn'
    ]
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                         check=True, capture_output=True)
    
    print("✅ Environment setup complete!")
    return True

def quick_start():
    """Quick start the bot mechanic"""
    print("\n" + "="*60)
    print("🚀 LOCAL BOT MECHANIC - QUICK START")
    print("="*60)
    
    # Setup environment
    if not setup_environment():
        return False
    
    # Import and run the mechanic
    sys.path.insert(0, str(Path('Intelligence/mechanic/local')))
    
    try:
        from bot_mechanic import LocalBotMechanic
        
        print("\n🧠 Starting Local Bot Mechanic...")
        mechanic = LocalBotMechanic()
        
        # Run initial scan
        print("🔍 Running initial health scan...")
        results = mechanic.quick_scan()
        
        if not results['healthy']:
            print(f"⚠️ Found {results['issues']} issues. Auto-fixing...")
            mechanic.auto_fix_all()
        
        print("✅ Bot Mechanic is ready!")
        print("\nChoose startup mode:")
        print("1. Run full scan and report")
        print("2. Start continuous monitoring")
        print("3. Generate dashboard")
        print("4. Auto-mode (scan + monitor)")
        
        choice = input("\nSelect mode (1-4): ").strip()
        
        if choice == '1':
            mechanic.deep_scan()
        elif choice == '2':
            mechanic.start_monitoring()
        elif choice == '3':
            report_path = mechanic.generate_html_report()
            print(f"📊 Dashboard: {report_path}")
        elif choice == '4':
            mechanic.deep_scan()
            print("\n🔄 Starting continuous monitoring...")
            mechanic.start_monitoring()
        
        return True
        
    except Exception as e:
        print(f"❌ Error starting mechanic: {e}")
        return False

if __name__ == "__main__":
    success = quick_start()
    if not success:
        print("\n❌ Quick start failed. Check the errors above.")
        sys.exit(1)
    else:
        print("\n✅ Local Bot Mechanic started successfully!")
