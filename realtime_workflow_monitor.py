#!/usr/bin/env python3
"""
Real-time Workflow Monitor
Watches for workflow execution evidence in real-time
"""

import time
import os
from datetime import datetime, timezone
import subprocess

def monitor_workflow_activity():
    """Monitor for signs of workflow execution"""
    
    print("🔍 REAL-TIME WORKFLOW MONITOR")
    print("=" * 50)
    print(f"Start time: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
    print()
    print("Watching for:")
    print("📊 01:25 UTC: Ultimate Data Collection Pipeline")
    print("📈 01:30 UTC: Multiple workflows")  
    print("🤖 01:45 UTC: Cloud Bot Mechanic")
    print()
    
    # Monitor for 15 minutes (until 01:31)
    start_time = time.time()
    last_git_check = 0
    last_file_check = 0
    
    while time.time() - start_time < 900:  # 15 minutes
        current_utc = datetime.now(timezone.utc)
        current_str = current_utc.strftime('%H:%M:%S UTC')
        
        # Check every minute
        if time.time() - last_git_check > 60:
            print(f"\n🕐 {current_str} - Checking for new activity...")
            
            # Check for new commits
            try:
                result = subprocess.run(['git', 'pull'], 
                                      capture_output=True, text=True, cwd='.')
                if 'Already up to date' not in result.stdout:
                    print("🎉 NEW GIT ACTIVITY DETECTED!")
                    print(f"   {result.stdout}")
                    
                    # Show recent commits
                    log_result = subprocess.run(['git', 'log', '--oneline', '-3'], 
                                              capture_output=True, text=True, cwd='.')
                    print("   Recent commits:")
                    for line in log_result.stdout.strip().split('\n'):
                        print(f"   📝 {line}")
                else:
                    print("   No new git activity")
            except:
                print("   Could not check git")
            
            last_git_check = time.time()
        
        # Check for file changes every 30 seconds
        if time.time() - last_file_check > 30:
            try:
                # Check Intelligence directory for new files
                intel_files = []
                for root, dirs, files in os.walk('Intelligence'):
                    for file in files:
                        filepath = os.path.join(root, file)
                        mtime = os.path.getmtime(filepath)
                        if time.time() - mtime < 120:  # Modified in last 2 minutes
                            intel_files.append((file, datetime.fromtimestamp(mtime, timezone.utc)))
                
                if intel_files:
                    print(f"   📁 Recent file activity ({len(intel_files)} files):")
                    for filename, mtime in sorted(intel_files, key=lambda x: x[1], reverse=True)[:3]:
                        print(f"      📄 {filename} at {mtime.strftime('%H:%M:%S UTC')}")
                
            except:
                pass
            
            last_file_check = time.time()
        
        # Special alerts for scheduled times
        minute = current_utc.minute
        hour = current_utc.hour
        
        if hour == 1 and minute == 25 and current_utc.second < 30:
            print(f"🚨 {current_str} - ULTIMATE DATA COLLECTION SHOULD TRIGGER NOW!")
        elif hour == 1 and minute == 30 and current_utc.second < 30:
            print(f"🚨 {current_str} - MULTIPLE WORKFLOWS SHOULD TRIGGER NOW!")
        elif hour == 1 and minute == 45 and current_utc.second < 30:
            print(f"🚨 {current_str} - CLOUD BOT MECHANIC SHOULD TRIGGER NOW!")
        
        time.sleep(10)  # Check every 10 seconds
    
    print(f"\n✅ Monitoring complete at {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")

if __name__ == "__main__":
    monitor_workflow_activity()
