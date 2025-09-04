#!/usr/bin/env python3
"""
Get detailed workflow failure information
"""
import requests
import json

def get_failure_details():
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

    # Get the latest failed run details
    url = 'https://api.github.com/repos/c-trading-bo/trading-bot-c-/actions/runs'
    response = requests.get(url, headers=headers, params={'per_page': 10})
    runs = response.json().get('workflow_runs', [])

    print('🔍 Latest Workflow Run Details:')
    print('=' * 50)

    failed_count = 0
    for run in runs:
        if run.get('conclusion') == 'failure' and failed_count < 3:
            failed_count += 1
            print(f'❌ FAILED: {run.get("name", "Unknown")}')
            print(f'   ID: {run.get("id")}')
            print(f'   Time: {run.get("created_at")}')
            print(f'   Commit: {run.get("head_sha", "")[:8]}')
            
            # Get job details
            jobs_url = f'https://api.github.com/repos/c-trading-bo/trading-bot-c-/actions/runs/{run.get("id")}/jobs'
            jobs_response = requests.get(jobs_url, headers=headers)
            if jobs_response.status_code == 200:
                jobs = jobs_response.json().get('jobs', [])
                for job in jobs:
                    if job.get('conclusion') == 'failure':
                        print(f'   📋 Job: {job.get("name", "Unknown")}')
                        print(f'   ⚠️  Steps with errors:')
                        for step in job.get('steps', []):
                            if step.get('conclusion') == 'failure':
                                print(f'     ❌ {step.get("name", "Unknown step")}')
            print()
    
    print(f'\n📊 Summary: Found {failed_count} recent failures')
    
    # Check AI Mechanic status
    ai_runs = [r for r in runs if 'Copilot' in r.get('name', '') or 'AI' in r.get('name', '')]
    if ai_runs:
        latest_ai = ai_runs[0]
        print(f'\n🧠 Latest AI Mechanic Run:')
        print(f'   Status: {latest_ai.get("conclusion", "unknown")}')
        print(f'   Time: {latest_ai.get("created_at")}')
        
        if latest_ai.get('conclusion') == 'failure':
            print('   ⚠️  AI Mechanic itself failed - checking logs...')
            # Get AI job details
            jobs_url = f'https://api.github.com/repos/c-trading-bo/trading-bot-c-/actions/runs/{latest_ai.get("id")}/jobs'
            jobs_response = requests.get(jobs_url, headers=headers)
            if jobs_response.status_code == 200:
                jobs = jobs_response.json().get('jobs', [])
                for job in jobs:
                    if job.get('conclusion') == 'failure':
                        print(f'     🔍 Failed job: {job.get("name")}')
    else:
        print('\n⚠️  No AI Mechanic runs found')

if __name__ == "__main__":
    get_failure_details()
