#!/usr/bin/env python3
"""
Test GitHub API connectivity with the AI Brain
"""
import os
import sys
import requests
import json
from datetime import datetime

def test_github_api():
    """Test GitHub API connectivity"""
    print("🧪 Testing GitHub API Connectivity...")
    
    # Load token from .env.github
    token = None
    try:
        with open('.env.github', 'r') as f:
            for line in f:
                if line.startswith('GITHUB_TOKEN='):
                    token = line.split('=')[1].strip()
                    break
    except Exception as e:
        print(f"❌ Failed to load token: {e}")
        return False
    
    if not token:
        print("❌ No GitHub token found")
        return False
    
    # Test API endpoints
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'Copilot-AI-Brain/2.0'
    }
    
    # Test 1: Get user info
    print("\n🔍 Test 1: User Authentication")
    try:
        response = requests.get('https://api.github.com/user', headers=headers)
        if response.status_code == 200:
            user_data = response.json()
            print(f"✅ Authenticated as: {user_data.get('login', 'Unknown')}")
        else:
            print(f"❌ Auth failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Auth error: {e}")
        return False
    
    # Test 2: Get repository info
    print("\n🔍 Test 2: Repository Access")
    try:
        repo_url = 'https://api.github.com/repos/c-trading-bo/trading-bot-c-'
        response = requests.get(repo_url, headers=headers)
        if response.status_code == 200:
            repo_data = response.json()
            print(f"✅ Repository: {repo_data.get('full_name', 'Unknown')}")
            print(f"✅ Default branch: {repo_data.get('default_branch', 'Unknown')}")
        else:
            print(f"❌ Repo access failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Repo error: {e}")
        return False
    
    # Test 3: Get workflow runs
    print("\n🔍 Test 3: Workflow Runs Access")
    try:
        workflows_url = 'https://api.github.com/repos/c-trading-bo/trading-bot-c-/actions/runs'
        response = requests.get(workflows_url, headers=headers, params={'per_page': 5})
        if response.status_code == 200:
            runs_data = response.json()
            print(f"✅ Found {runs_data.get('total_count', 0)} workflow runs")
            
            # Show recent runs
            for run in runs_data.get('workflow_runs', [])[:3]:
                status = run.get('status', 'unknown')
                conclusion = run.get('conclusion', 'unknown')
                name = run.get('name', 'Unknown')
                print(f"   📊 {name}: {status}/{conclusion}")
        else:
            print(f"❌ Workflow access failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Workflow error: {e}")
        return False
    
    # Test 4: Rate limit check
    print("\n🔍 Test 4: Rate Limit Status")
    try:
        rate_url = 'https://api.github.com/rate_limit'
        response = requests.get(rate_url, headers=headers)
        if response.status_code == 200:
            rate_data = response.json()
            core_limit = rate_data.get('resources', {}).get('core', {})
            remaining = core_limit.get('remaining', 0)
            total = core_limit.get('limit', 0)
            print(f"✅ Rate limit: {remaining}/{total} remaining")
        else:
            print(f"❌ Rate limit check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Rate limit error: {e}")
    
    print("\n🎉 GitHub API connectivity test completed!")
    return True

def test_ai_brain_with_api():
    """Test the AI Brain with real API access"""
    print("\n🧠 Testing AI Brain with GitHub API...")
    
    # Set environment variables
    os.environ['GITHUB_TOKEN'] = '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'
    os.environ['GITHUB_REPOSITORY'] = 'c-trading-bo/trading-bot-c-'
    os.environ['GITHUB_RUN_ID'] = '123456789'
    
    try:
        # Import and run AI Brain
        sys.path.append('.github/copilot_mechanic')
        from copilot_ai_brain import CopilotEnterpriseAIBrain
        
        # Initialize AI Brain
        brain = CopilotEnterpriseAIBrain()
        
        # Test workflow analysis
        print("🔍 Testing workflow analysis...")
        sample_workflow = """
name: Test Workflow
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: echo "test"
"""
        
        analysis = brain.analyze_workflow_failure(sample_workflow, "Sample analysis")
        print(f"✅ Analysis generated: {len(analysis)} characters")
        
        # Test knowledge system
        print("🔍 Testing knowledge system...")
        test_workflow_run = {"name": "test_workflow"}
        test_fix_data = {
            "root_cause": "workflow_syntax_error",
            "fix_type": "yaml_edit", 
            "fix_code": "sample fix code",
            "confidence": 95
        }
        brain.learn_fix(test_workflow_run, test_fix_data)
        knowledge = brain.knowledge_base.get('learned_fixes', {})
        print(f"✅ Knowledge entries: {len(knowledge)}")
        
        print("🎉 AI Brain test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ AI Brain test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 COPILOT ENTERPRISE AI BRAIN - API TEST")
    print("=" * 60)
    
    # Test GitHub API
    api_success = test_github_api()
    
    if api_success:
        # Test AI Brain
        brain_success = test_ai_brain_with_api()
        
        if brain_success:
            print("\n✅ ALL TESTS PASSED! 🎉")
            print("🚀 Your Copilot Enterprise AI Brain is ready for production!")
        else:
            print("\n⚠️ API works but AI Brain needs attention")
    else:
        print("\n❌ GitHub API test failed - check your token")
    
    print("\n" + "=" * 60)
