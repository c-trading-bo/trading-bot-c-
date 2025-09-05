#!/usr/bin/env python3
"""
Real-time GitHub Workflow Monitor
Monitors workflow executions and reports if schedules are working
"""

import requests
import time
import json
from datetime import datetime, timezone

# GitHub API setup
REPO_OWNER = "c-trading-bo"
REPO_NAME = "trading-bot-c-"
API_BASE = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"

# Expected workflows to trigger soon
EXPECTED_WORKFLOWS = {
    "21:00": ["ES/NQ Critical Trading (Team)", "🌩️🧠 Ultimate AI+Cloud Bot Mechanic - Enterprise Defense System"],
    "21:15": ["Ultimate ML/RL/Intel System (Team Optimized)"],
    "21:10": ["ES/NQ Critical Trading (Team)"],
    "21:20": ["ES/NQ Critical Trading (Team)"],
    "21:30": ["ES/NQ Critical Trading (Team)"],
    "21:45": ["🌩️🧠 Ultimate AI+Cloud Bot Mechanic - Enterprise Defense System"]
}

def get_recent_workflow_runs():
    """Get recent workflow runs from GitHub API"""
    try:
        # Check if we have GITHUB_TOKEN in environment
        import os
        token = os.environ.get('GITHUB_TOKEN')
        
        headers = {}
        if token:
            headers['Authorization'] = f'token {token}'
            headers['Accept'] = 'application/vnd.github.v3+json'
        
        url = f"{API_BASE}/actions/runs"
        params = {
            'per_page': 20,
            'status': 'completed,in_progress,queued'
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ GitHub API Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error fetching workflow runs: {e}")
        return None

def monitor_workflows(duration_minutes=30):
    """Monitor workflows for the specified duration"""
    print("🔍 STARTING REAL-TIME WORKFLOW MONITORING")
    print("=" * 60)
    print(f"⏰ Monitoring for {duration_minutes} minutes...")
    print(f"📅 Start time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()
    
    start_time = time.time()
    last_check = {}
    
    while time.time() - start_time < duration_minutes * 60:
        current_time = datetime.now(timezone.utc)
        time_str = current_time.strftime("%H:%M")
        
        print(f"⏰ {current_time.strftime('%H:%M:%S UTC')} - Checking for new workflow runs...")
        
        # Get recent runs
        runs_data = get_recent_workflow_runs()
        
        if runs_data and 'workflow_runs' in runs_data:
            recent_runs = []
            cutoff_time = time.time() - 300  # Last 5 minutes
            
            for run in runs_data['workflow_runs']:
                created_at = datetime.fromisoformat(run['created_at'].replace('Z', '+00:00'))
                if created_at.timestamp() > cutoff_time:
                    recent_runs.append(run)
            
            if recent_runs:
                print(f"🚀 Found {len(recent_runs)} recent workflow runs:")
                for run in recent_runs:
                    workflow_name = run['name']
                    status = run['status']
                    conclusion = run['conclusion']
                    created_at = datetime.fromisoformat(run['created_at'].replace('Z', '+00:00'))
                    
                    status_emoji = {
                        'queued': '⏳',
                        'in_progress': '🔄', 
                        'completed': '✅' if conclusion == 'success' else '❌'
                    }.get(status, '❓')
                    
                    print(f"  {status_emoji} {workflow_name}")
                    print(f"     Status: {status} | Created: {created_at.strftime('%H:%M:%S UTC')}")
                    
                    # Check if this matches expected schedule
                    expected_time = time_str
                    if expected_time in EXPECTED_WORKFLOWS:
                        if workflow_name in EXPECTED_WORKFLOWS[expected_time]:
                            print(f"     ✅ SCHEDULE VERIFIED: Expected at {expected_time} UTC")
                    print()
            else:
                print("📭 No recent workflow runs in last 5 minutes")
        else:
            print("❌ Could not fetch workflow data")
        
        print("-" * 40)
        time.sleep(60)  # Check every minute
    
    print("🏁 MONITORING COMPLETE")

if __name__ == "__main__":
    monitor_workflows(30)  # Monitor for 30 minutes
