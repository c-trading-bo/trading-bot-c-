#!/usr/bin/env python3
"""
Cloud Bot Mechanic Status Checker
Check if the Cloud Bot Mechanic is working and responding to failures
"""

import os
import json
from datetime import datetime, timezone

def check_mechanic_status():
    """Check Cloud Bot Mechanic activity and status"""
    
    print("🤖 CLOUD BOT MECHANIC STATUS CHECK")
    print("=" * 50)
    print(f"Current Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()
    
    # Check for mechanic data directory
    mechanic_dir = "Intelligence/data/mechanic"
    if os.path.exists(mechanic_dir):
        print(f"✅ Mechanic directory exists: {mechanic_dir}")
        
        # List all files
        files = os.listdir(mechanic_dir)
        print(f"📁 Found {len(files)} mechanic files:")
        
        for file in sorted(files):
            filepath = os.path.join(mechanic_dir, file)
            mtime = os.path.getmtime(filepath)
            mtime_str = datetime.fromtimestamp(mtime, timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            size = os.path.getsize(filepath)
            print(f"   📄 {file} ({size} bytes) - {mtime_str}")
            
            # Show recent files content
            if datetime.fromtimestamp(mtime, timezone.utc) > datetime.now(timezone.utc).replace(hour=0, minute=0, second=0):
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        if content.strip():
                            print(f"      📝 Content preview: {content[:100]}...")
                except:
                    pass
    else:
        print(f"❌ Mechanic directory not found: {mechanic_dir}")
    
    print()
    
    # Check Cloud Bot Mechanic schedule
    print("⏰ CLOUD BOT MECHANIC SCHEDULE:")
    print("   Schedule: */45 * * * * (Every 45 minutes)")
    
    current_time = datetime.now(timezone.utc)
    current_minute = current_time.minute
    
    # Calculate trigger times
    if current_minute < 45:
        last_trigger = current_time.replace(minute=0, second=0, microsecond=0)
        next_trigger = current_time.replace(minute=45, second=0, microsecond=0)
    else:
        last_trigger = current_time.replace(minute=45, second=0, microsecond=0)
        next_trigger = current_time.replace(hour=current_time.hour+1, minute=0, second=0, microsecond=0)
    
    print(f"   Last trigger: {last_trigger.strftime('%H:%M UTC')} ({int((current_time - last_trigger).total_seconds() / 60)} min ago)")
    print(f"   Next trigger: {next_trigger.strftime('%H:%M UTC')} (in {int((next_trigger - current_time).total_seconds() / 60)} min)")
    
    # Check if mechanic should have run recently
    minutes_since_last = int((current_time - last_trigger).total_seconds() / 60)
    if minutes_since_last < 5:
        print(f"   🚨 MECHANIC SHOULD BE RUNNING NOW! (triggered {minutes_since_last} min ago)")
    elif minutes_since_last < 15:
        print(f"   ⚠️  Mechanic should have run recently ({minutes_since_last} min ago)")
    else:
        print(f"   ✅ Mechanic timing normal ({minutes_since_last} min since last trigger)")
    
    print()
    
    # Check for workflow failures that should trigger mechanic
    print("🔍 CHECKING FOR WORKFLOW FAILURES:")
    
    # Look for any failure indicators in recent commits
    try:
        import subprocess
        result = subprocess.run(['git', 'log', '--oneline', '--since=2 hours ago'], 
                              capture_output=True, text=True, cwd='.')
        commits = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        failure_indicators = []
        success_indicators = []
        
        for commit in commits:
            commit_lower = commit.lower()
            if any(word in commit_lower for word in ['fail', 'error', 'fix', 'emergency']):
                failure_indicators.append(commit)
            elif any(word in commit_lower for word in ['success', 'complete', 'update']):
                success_indicators.append(commit)
        
        print(f"   📊 Found {len(commits)} recent commits")
        
        if failure_indicators:
            print(f"   ⚠️  Potential failure indicators ({len(failure_indicators)}):")
            for commit in failure_indicators[:3]:
                print(f"      🔴 {commit}")
        
        if success_indicators:
            print(f"   ✅ Success indicators ({len(success_indicators)}):")
            for commit in success_indicators[:3]:
                print(f"      🟢 {commit}")
                
    except Exception as e:
        print(f"   ❌ Could not check git history: {e}")
    
    print()
    
    # Summary and recommendations
    print("🎯 MECHANIC STATUS SUMMARY:")
    
    if os.path.exists(mechanic_dir) and os.listdir(mechanic_dir):
        print("   ✅ Mechanic data directory exists with files")
    else:
        print("   ❌ No mechanic activity detected")
    
    if minutes_since_last < 15:
        print("   ⚠️  Mechanic should have run recently - check GitHub Actions")
    
    print()
    print("🛠️  TROUBLESHOOTING STEPS:")
    print("1. Check GitHub Actions tab for Cloud Bot Mechanic runs")
    print("2. Look for workflow failures that should trigger mechanic")
    print("3. Verify mechanic workflow is not disabled")
    print("4. Check repository Actions permissions")
    print("5. Manually trigger mechanic with workflow_dispatch")
    
    return {
        'mechanic_dir_exists': os.path.exists(mechanic_dir),
        'minutes_since_trigger': minutes_since_last,
        'next_trigger_minutes': int((next_trigger - current_time).total_seconds() / 60),
        'should_be_active': minutes_since_last < 5
    }

if __name__ == "__main__":
    status = check_mechanic_status()
    print(f"\n📊 Status: {json.dumps(status, indent=2)}")
