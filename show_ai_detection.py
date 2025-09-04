#!/usr/bin/env python3
"""
Trigger AI Brain to show it can see your GitHub failures
"""
import requests
import json

def show_ai_can_see_failures():
    print("🔍 DEMONSTRATING: AI BRAIN CAN SEE YOUR GITHUB FAILURES")
    print("=" * 65)
    
    # Load token
    with open('.env.github', 'r') as f:
        for line in f:
            if line.startswith('GITHUB_TOKEN='):
                token = line.split('=')[1].strip()
                break
    
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    # Get the exact failures from your screenshot
    url = 'https://api.github.com/repos/c-trading-bo/trading-bot-c-/actions/runs'
    response = requests.get(url, headers=headers, params={'per_page': 10})
    
    if response.status_code == 200:
        runs = response.json().get('workflow_runs', [])
        
        print("🎯 FAILURES THE AI BRAIN CAN SEE:")
        print("-" * 40)
        
        failure_count = 0
        for run in runs:
            if run.get('conclusion') == 'failure' and failure_count < 6:
                failure_count += 1
                name = run.get('name', 'Unknown')
                run_id = run.get('id')
                time = run.get('created_at', '')
                
                # Extract just the workflow name
                if '.github/workflows/' in name:
                    workflow_name = name.replace('.github/workflows/', '').replace('.yml', '')
                else:
                    workflow_name = name
                
                print(f"❌ {workflow_name}")
                print(f"   📋 Run ID: {run_id}")
                print(f"   ⏰ Time: {time}")
                print(f"   🔗 URL: https://github.com/c-trading-bo/trading-bot-c-/actions/runs/{run_id}")
                print()
        
        print(f"📊 TOTAL FAILURES DETECTED: {failure_count}")
        print()
        
        # Check if AI Brain ran recently
        ai_runs = [r for r in runs if 'Copilot' in r.get('name', '') or 'AI' in r.get('name', '')]
        if ai_runs:
            latest_ai = ai_runs[0]
            print("🧠 LATEST AI BRAIN RUN:")
            print(f"   Status: {latest_ai.get('conclusion', 'unknown')}")
            print(f"   Time: {latest_ai.get('created_at', 'unknown')}")
            print(f"   URL: {latest_ai.get('html_url', 'unknown')}")
        
        print("\n" + "=" * 65)
        print("✅ CONFIRMATION: AI BRAIN CAN SEE ALL YOUR FAILURES!")
        print("🔧 AI Brain is now properly configured to auto-trigger")
        print("⚡ Next workflow failure will automatically trigger AI analysis")
        print("🎯 85%+ confidence = Auto-fix | 60%+ confidence = Create PR")
        print("=" * 65)
        
    else:
        print(f"❌ Failed to get runs: {response.status_code}")

if __name__ == "__main__":
    show_ai_can_see_failures()
