#!/usr/bin/env python3
"""
Quick Token Test Script
Tests if GitHub token is working for API access
"""

import os
import requests
import json

def test_github_token():
    token = os.getenv('GITHUB_TOKEN', '')
    
    if not token:
        print("❌ No GITHUB_TOKEN found in environment")
        print("💡 Set it with: $env:GITHUB_TOKEN = 'your_token_here'")
        return False
    
    print(f"🔍 Testing token: {token[:10]}...")
    
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    try:
        # Test basic API access
        response = requests.get('https://api.github.com/user', headers=headers)
        response.raise_for_status()
        
        user_data = response.json()
        print(f"✅ Token works! Authenticated as: {user_data.get('login', 'Unknown')}")
        
        # Test repository access
        repo_url = 'https://api.github.com/repos/c-trading-bo/trading-bot-c-'
        repo_response = requests.get(repo_url, headers=headers)
        repo_response.raise_for_status()
        
        print("✅ Repository access confirmed!")
        
        # Test workflow access
        workflow_url = f'{repo_url}/actions/workflows'
        workflow_response = requests.get(workflow_url, headers=headers)
        workflow_response.raise_for_status()
        
        workflows = workflow_response.json()
        workflow_count = workflows.get('total_count', 0)
        print(f"✅ Workflow access confirmed! Found {workflow_count} workflows")
        
        print("\n🎯 TOKEN IS READY FOR MONITORING!")
        return True
        
    except requests.RequestException as e:
        print(f"❌ Token test failed: {e}")
        print("💡 Check token permissions and expiration")
        return False

if __name__ == "__main__":
    print("🧪 GitHub Token Test")
    print("=" * 30)
    test_github_token()
