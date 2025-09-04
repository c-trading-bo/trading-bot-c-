#!/usr/bin/env python3
"""
Check current GitHub Actions workflow status
"""
import requests
import json
from datetime import datetime

def check_github_actions():
    print("🔍 Checking GitHub Actions Status...")
    
    # Load token
    token = None
    try:
        with open('.env.github', 'r') as f:
            for line in f:
                if line.startswith('GITHUB_TOKEN='):
                    token = line.split('=')[1].strip()
                    break
    except Exception as e:
        print(f"❌ Failed to load token: {e}")
        return
    
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'Copilot-AI-Brain/2.0'
    }
    
    # Get recent workflow runs
    url = 'https://api.github.com/repos/c-trading-bo/trading-bot-c-/actions/runs'
    response = requests.get(url, headers=headers, params={'per_page': 10})
    
    if response.status_code == 200:
        data = response.json()
        runs = data.get('workflow_runs', [])
        
        print(f"\n📊 Recent Workflow Runs ({len(runs)} shown):")
        print("-" * 80)
        
        for run in runs:
            name = run.get('name', 'Unknown')
            status = run.get('status', 'unknown')
            conclusion = run.get('conclusion', 'unknown')
            created = run.get('created_at', '')
            
            # Format time
            try:
                dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                time_str = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
            except:
                time_str = created
            
            # Status icon
            if conclusion == 'success':
                icon = '✅'
            elif conclusion == 'failure':
                icon = '❌' 
            elif status == 'in_progress':
                icon = '🔄'
            else:
                icon = '⚪'
            
            print(f"{icon} {name}")
            print(f"   Status: {status}/{conclusion}")
            print(f"   Time: {time_str}")
            print()
        
        # Check for AI Mechanic runs
        ai_runs = [r for r in runs if 'Copilot' in r.get('name', '') or 'AI' in r.get('name', '')]
        if ai_runs:
            print("🧠 AI Mechanic Activity:")
            for run in ai_runs[:3]:
                print(f"   📊 {run.get('conclusion', 'unknown')} - {run.get('created_at', '')}")
        else:
            print("⚠️ No recent AI Mechanic runs found")
        
        # Check for failures that need fixing
        failed_runs = [r for r in runs if r.get('conclusion') == 'failure']
        if failed_runs:
            print(f"\n🔧 {len(failed_runs)} Recent Failures Detected:")
            for run in failed_runs[:3]:
                print(f"   ❌ {run.get('name', 'Unknown')} - {run.get('created_at', '')}")
            print("   💡 AI Brain should automatically analyze and fix these!")
        
    else:
        print(f"❌ Failed to get workflow runs: {response.status_code}")

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 GITHUB ACTIONS STATUS CHECK")
    print("=" * 60)
    check_github_actions()
    print("=" * 60)
