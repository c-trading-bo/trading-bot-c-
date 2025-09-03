#!/usr/bin/env python3
"""
Quick Mechanic Status Checker
"""

import json
from pathlib import Path

def quick_status():
    print("🤖 MECHANIC QUICK STATUS CHECK")
    print("=" * 40)
    
    db_path = Path("Intelligence/mechanic/database")
    
    # Check if mechanic is running
    if db_path.exists():
        print("✅ Mechanic database found")
        
        # Check dashboard status
        dashboard_file = db_path / "dashboard_status.json"
        if dashboard_file.exists():
            try:
                with open(dashboard_file) as f:
                    status = json.load(f)
                
                health = "✅ HEALTHY" if status.get('is_healthy', False) else "⚠️ ISSUES DETECTED"
                print(f"🏥 Health: {health}")
                print(f"📊 Health Score: {status.get('health_score', 0)}%")
                print(f"🔍 Issues Count: {status.get('issues_count', 0)}")
                print(f"📁 Files Monitored: {status.get('files_count', 0):,}")
                print(f"🧠 Features Found: {status.get('feature_count', 0):,}")
                print(f"⏰ Last Update: {status.get('timestamp', 'Unknown')}")
                
            except Exception as e:
                print(f"❌ Error reading status: {e}")
        else:
            print("⚠️ No dashboard status file found")
        
        # Check repairs
        repairs_file = db_path / "repairs.json"
        if repairs_file.exists():
            try:
                with open(repairs_file) as f:
                    repairs = json.load(f)
                print(f"🔧 Total Repairs Made: {len(repairs)}")
                if repairs:
                    latest = repairs[-1]
                    print(f"🕐 Latest Repair: {latest.get('system')} - {latest.get('fix_type')}")
            except Exception as e:
                print(f"❌ Error reading repairs: {e}")
        
        # Check if running
        files = list(db_path.glob("*.json"))
        if files:
            print(f"✅ Mechanic appears to be ACTIVE ({len(files)} database files)")
        else:
            print("⚠️ Mechanic may not be running")
            
    else:
        print("❌ Mechanic database not found - mechanic not running")
    
    print("\n🎯 TO MONITOR LIVE:")
    print("   python monitor_mechanic_live.py")
    print("\n🎯 TO START MECHANIC:")
    print("   python auto_background_mechanic.py")

if __name__ == "__main__":
    quick_status()
