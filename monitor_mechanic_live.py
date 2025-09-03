#!/usr/bin/env python3
"""
REAL-TIME MECHANIC MONITOR
Shows live status of what the mechanic is doing
"""

import json
import time
import os
from datetime import datetime
from pathlib import Path

def monitor_mechanic_activity():
    """Monitor and display real-time mechanic activity"""
    print("🔍 REAL-TIME MECHANIC MONITOR")
    print("=" * 50)
    print("Press Ctrl+C to stop monitoring")
    print()
    
    base_path = Path("Intelligence/mechanic/database")
    last_check = {}
    
    try:
        while True:
            # Clear screen on Windows
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("🤖 AUTO-BACKGROUND MECHANIC - LIVE STATUS")
            print("=" * 60)
            print(f"⏰ Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # Check dashboard status
            try:
                dashboard_file = base_path / "dashboard_status.json"
                if dashboard_file.exists():
                    with open(dashboard_file) as f:
                        status = json.load(f)
                    
                    health_emoji = "✅" if status.get('is_healthy', False) else "⚠️"
                    print(f"{health_emoji} HEALTH STATUS:")
                    print(f"   Health Score: {status.get('health_score', 0)}%")
                    print(f"   Issues Count: {status.get('issues_count', 0)}")
                    print(f"   Files Monitored: {status.get('files_count', 0):,}")
                    print(f"   Features Detected: {status.get('feature_count', 0):,}")
                    print()
            except Exception as e:
                print(f"⚠️ Dashboard status unavailable: {e}")
            
            # Check recent repairs
            try:
                repairs_file = base_path / "repairs.json"
                if repairs_file.exists():
                    with open(repairs_file) as f:
                        repairs = json.load(f)
                    
                    print("🔧 RECENT AUTO-REPAIRS:")
                    if repairs:
                        for repair in repairs[-5:]:  # Last 5 repairs
                            timestamp = repair.get('timestamp', 'Unknown')
                            system = repair.get('system', 'Unknown')
                            fix_type = repair.get('fix_type', 'Unknown')
                            success = "✅" if repair.get('success', False) else "❌"
                            print(f"   {success} {timestamp[:19]} - {system}: {fix_type}")
                    else:
                        print("   No repairs needed yet")
                    print()
            except Exception as e:
                print(f"⚠️ Repairs log unavailable: {e}")
            
            # Check features database
            try:
                features_file = base_path / "features.json"
                if features_file.exists():
                    with open(features_file) as f:
                        features = json.load(f)
                    
                    print("🧠 SYSTEM INTELLIGENCE:")
                    if isinstance(features, dict):
                        for category, items in list(features.items())[:8]:  # First 8 categories
                            if isinstance(items, list):
                                print(f"   🔍 {category:20} {len(items):4} items")
                    print()
            except Exception as e:
                print(f"⚠️ Features database unavailable: {e}")
            
            # Monitor file changes
            current_files = {}
            for file_path in base_path.glob("*.json"):
                try:
                    stat = file_path.stat()
                    current_files[file_path.name] = stat.st_mtime
                except:
                    continue
            
            # Show file activity
            print("📁 FILE ACTIVITY:")
            for filename, mtime in current_files.items():
                age_seconds = time.time() - mtime
                if age_seconds < 300:  # Modified in last 5 minutes
                    age_str = f"{int(age_seconds)}s ago"
                    status_emoji = "🔥" if age_seconds < 60 else "📝"
                else:
                    age_minutes = int(age_seconds / 60)
                    age_str = f"{age_minutes}m ago"
                    status_emoji = "📄"
                
                print(f"   {status_emoji} {filename:25} {age_str}")
            
            print()
            print("🔄 Next update in 10 seconds...")
            print("Press Ctrl+C to stop monitoring")
            
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")
    except Exception as e:
        print(f"\n❌ Monitor error: {e}")

if __name__ == "__main__":
    monitor_mechanic_activity()
