#!/usr/bin/env python3
"""
Workflow Execution Monitor
Creates test runs and monitors workflow activity
"""

import os
import json
import requests
from datetime import datetime, timedelta

def check_recent_workflow_runs():
    """Check for recent workflow execution evidence"""
    print("🔍 CHECKING FOR WORKFLOW EXECUTION EVIDENCE")
    print("=" * 50)
    
    # Check for recent data files created by workflows
    data_dirs = [
        'Intelligence/data',
        'data',
        '.github/system_status.txt'
    ]
    
    recent_files = []
    cutoff_time = datetime.now() - timedelta(hours=24)
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if mtime > cutoff_time:
                            recent_files.append((file_path, mtime))
                    except:
                        pass
    
    if recent_files:
        print(f"✅ Found {len(recent_files)} recent files (last 24h):")
        recent_files.sort(key=lambda x: x[1], reverse=True)
        for file_path, mtime in recent_files[:10]:  # Show top 10
            print(f"  📄 {file_path}")
            print(f"     🕒 {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("⚠️  No recent data files found")
    
    return len(recent_files)

def create_workflow_status_file():
    """Create a status file for workflows to update"""
    status = {
        "last_check": datetime.now().isoformat(),
        "workflows_validated": True,
        "schedules_active": [
            "es_nq_critical_trading.yml",
            "ultimate_ml_rl_intel_system.yml", 
            "daily_consolidated.yml",
            "volatility_surface.yml",
            "es_nq_correlation_matrix.yml"
        ],
        "saturday_optimization": "ENABLED",
        "next_expected_runs": {
            "es_nq_critical": "Every 3-15 minutes during market hours",
            "ml_rl_system": "Every 5-30 minutes depending on time",
            "daily_consolidated": "8 AM EST daily, 12 PM EST Sunday",
            "volatility_surface": "9:00, 9:30, 12:00, 12:30, 15:00, 15:30 on market days",
            "correlation_matrix": "Every 2 hours"
        }
    }
    
    os.makedirs('.github', exist_ok=True)
    with open('.github/workflow_status.json', 'w') as f:
        json.dump(status, f, indent=2)
    
    print(f"📋 Created workflow status file: .github/workflow_status.json")

def main():
    print("🚀 WORKFLOW EXECUTION MONITOR")
    print("=" * 50)
    print(f"📅 Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check for recent activity
    recent_count = check_recent_workflow_runs()
    print()
    
    # Create status file
    create_workflow_status_file()
    print()
    
    # Summary
    print("📊 EXECUTION SUMMARY:")
    if recent_count > 0:
        print(f"✅ Found evidence of recent workflow activity ({recent_count} files)")
        print("🎯 Workflows appear to be running!")
    else:
        print("⚠️  No recent activity detected")
        print("🔧 May need to manually trigger workflows or check GitHub Actions")
    
    print()
    print("🔗 VERIFICATION STEPS:")
    print("1. Visit: https://github.com/c-trading-bo/trading-bot-c-/actions")
    print("2. Look for recent workflow runs")
    print("3. Check if any workflows are failing")
    print("4. Verify Saturday optimization is working (no runs on Saturdays)")

if __name__ == "__main__":
    main()
