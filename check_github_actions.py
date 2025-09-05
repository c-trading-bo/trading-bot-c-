#!/usr/bin/env python3
"""
GitHub Actions Status Checker
Checks if workflows are actually running in GitHub Actions tab
"""

import requests
import json
import os
from datetime import datetime, timezone

def check_github_actions():
    """Check GitHub Actions status"""
    
    # Get repository info
    owner = "c-trading-bo"  # From your repo context
    repo = "trading-bot-c-"
    
    print("🔍 CHECKING GITHUB ACTIONS STATUS")
    print("=" * 50)
    print(f"Repository: {owner}/{repo}")
    print(f"Current Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()
    
    # GitHub API endpoints
    base_url = f"https://api.github.com/repos/{owner}/{repo}"
    workflows_url = f"{base_url}/actions/workflows"
    runs_url = f"{base_url}/actions/runs"
    
    # Headers for GitHub API
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "WorkflowChecker/1.0"
    }
    
    # Add token if available
    github_token = os.environ.get('GITHUB_TOKEN')
    if github_token:
        headers["Authorization"] = f"token {github_token}"
        print("✅ Using GitHub token for authentication")
    else:
        print("⚠️  No GitHub token - using anonymous access (rate limited)")
    
    print()
    
    try:
        # Check workflows
        print("📋 CHECKING AVAILABLE WORKFLOWS:")
        print("-" * 30)
        
        response = requests.get(workflows_url, headers=headers)
        print(f"Workflows API Status: {response.status_code}")
        
        if response.status_code == 200:
            workflows_data = response.json()
            workflows = workflows_data.get('workflows', [])
            
            print(f"Found {len(workflows)} workflows:")
            
            ultimate_workflows = []
            regular_workflows = []
            
            for workflow in workflows:
                name = workflow.get('name', 'Unknown')
                state = workflow.get('state', 'unknown')
                path = workflow.get('path', '')
                
                if 'Ultimate' in name or 'ultimate' in path:
                    ultimate_workflows.append((name, state, path))
                else:
                    regular_workflows.append((name, state, path))
            
            print(f"\n🚀 ULTIMATE WORKFLOWS ({len(ultimate_workflows)}):")
            for name, state, path in ultimate_workflows:
                status_icon = "✅" if state == "active" else "❌"
                print(f"  {status_icon} {name} ({state})")
                print(f"      Path: {path}")
            
            print(f"\n📊 REGULAR WORKFLOWS ({len(regular_workflows)}):")
            for name, state, path in regular_workflows[:10]:  # Show first 10
                status_icon = "✅" if state == "active" else "❌"
                print(f"  {status_icon} {name} ({state})")
            
            if len(regular_workflows) > 10:
                print(f"  ... and {len(regular_workflows) - 10} more")
                
        else:
            print(f"❌ Error accessing workflows: {response.status_code}")
            print(f"Response: {response.text[:200]}")
        
        print("\n" + "=" * 50)
        
        # Check recent workflow runs
        print("🏃 CHECKING RECENT WORKFLOW RUNS:")
        print("-" * 30)
        
        runs_response = requests.get(runs_url, headers=headers, params={"per_page": 20})
        print(f"Runs API Status: {runs_response.status_code}")
        
        if runs_response.status_code == 200:
            runs_data = runs_response.json()
            runs = runs_data.get('workflow_runs', [])
            
            print(f"Found {len(runs)} recent runs:")
            print()
            
            ultimate_runs = []
            recent_runs = []
            
            for run in runs:
                name = run.get('name', 'Unknown')
                status = run.get('status', 'unknown')
                conclusion = run.get('conclusion', 'unknown')
                created_at = run.get('created_at', '')
                updated_at = run.get('updated_at', '')
                
                run_info = {
                    'name': name,
                    'status': status,
                    'conclusion': conclusion,
                    'created_at': created_at,
                    'updated_at': updated_at,
                    'is_ultimate': 'Ultimate' in name or 'ultimate' in name.lower()
                }
                
                if run_info['is_ultimate']:
                    ultimate_runs.append(run_info)
                recent_runs.append(run_info)
            
            # Show Ultimate workflow runs
            if ultimate_runs:
                print(f"🚀 ULTIMATE WORKFLOW RUNS ({len(ultimate_runs)}):")
                for run in ultimate_runs[:5]:
                    status_icon = get_status_icon(run['status'], run['conclusion'])
                    created_time = parse_github_time(run['created_at'])
                    print(f"  {status_icon} {run['name']}")
                    print(f"      Status: {run['status']} | Conclusion: {run['conclusion']}")
                    print(f"      Created: {created_time}")
                    print()
            else:
                print("❌ No Ultimate workflow runs found!")
            
            # Show all recent runs
            print("📋 ALL RECENT RUNS:")
            for run in recent_runs[:10]:
                status_icon = get_status_icon(run['status'], run['conclusion'])
                created_time = parse_github_time(run['created_at'])
                print(f"  {status_icon} {run['name'][:50]}")
                print(f"      {run['status']}/{run['conclusion']} at {created_time}")
                
        else:
            print(f"❌ Error accessing runs: {runs_response.status_code}")
            print(f"Response: {runs_response.text[:200]}")
            
    except Exception as e:
        print(f"❌ Error checking GitHub Actions: {e}")
        import traceback
        traceback.print_exc()

def get_status_icon(status, conclusion):
    """Get appropriate icon for workflow status"""
    if status == "completed":
        if conclusion == "success":
            return "✅"
        elif conclusion == "failure":
            return "❌"
        elif conclusion == "cancelled":
            return "⏹️"
        else:
            return "⚠️"
    elif status == "in_progress":
        return "🔄"
    elif status == "queued":
        return "⏳"
    else:
        return "❓"

def parse_github_time(time_str):
    """Parse GitHub timestamp"""
    try:
        dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S UTC')
    except:
        return time_str

if __name__ == "__main__":
    check_github_actions()
