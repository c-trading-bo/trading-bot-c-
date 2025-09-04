#!/usr/bin/env python3
"""
Manual AI Brain Analysis - Check if AI can see GitHub failures
"""
import os
import sys
import requests
import json
from datetime import datetime

# Set environment for AI Brain
os.environ['GITHUB_TOKEN'] = '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'
os.environ['GITHUB_REPOSITORY'] = 'c-trading-bo/trading-bot-c-'

def test_ai_brain_detection():
    print("🧠 Testing AI Brain Detection of Failures...")
    print("=" * 60)
    
    # Import AI Brain
    sys.path.append('.github/copilot_mechanic')
    from copilot_ai_brain import CopilotEnterpriseAIBrain
    
    # Initialize AI Brain
    brain = CopilotEnterpriseAIBrain()
    
    # Test 1: Can AI see the workflow runs?
    print("🔍 Test 1: Can AI Brain detect workflow failures?")
    try:
        recent_runs = brain.get_recent_workflow_runs()
        print(f"✅ AI detected {len(recent_runs)} recent workflow runs")
        
        failed_runs = [r for r in recent_runs if r.get('conclusion') == 'failure']
        print(f"🔥 AI found {len(failed_runs)} failed runs")
        
        for run in failed_runs[:3]:
            print(f"   ❌ {run.get('name', 'Unknown')} - {run.get('created_at', 'Unknown time')}")
    
    except Exception as e:
        print(f"❌ AI Brain failed to detect runs: {e}")
        return False
    
    # Test 2: Can AI analyze specific failure?
    print(f"\n🔍 Test 2: Can AI Brain analyze specific failure?")
    if failed_runs:
        try:
            sample_failure = failed_runs[0]
            print(f"📋 Analyzing: {sample_failure.get('name', 'Unknown')}")
            
            # Get workflow content
            workflow_name = sample_failure.get('name', '').replace('.github/workflows/', '').replace('.yml', '')
            workflow_path = f".github/workflows/{workflow_name}.yml"
            
            if os.path.exists(workflow_path):
                with open(workflow_path, 'r') as f:
                    workflow_content = f.read()
                
                print(f"✅ Loaded workflow file: {len(workflow_content)} characters")
                
                # AI Analysis
                analysis = brain.analyze_workflow_failure(workflow_content, f"Failure in {workflow_name}")
                print(f"✅ AI generated analysis: {len(analysis)} characters")
                print(f"📊 Analysis preview: {analysis[:200]}...")
                
                # Check if AI can generate fixes
                print(f"\n🔍 Test 3: Can AI generate fixes?")
                fixes = brain.extract_fixes(analysis)
                if fixes:
                    print(f"✅ AI generated {len(fixes)} potential fixes")
                    for i, fix in enumerate(fixes[:2]):
                        print(f"   🔧 Fix {i+1}: {fix.get('fix_type', 'unknown')} (confidence: {fix.get('confidence', 0)}%)")
                else:
                    print("⚠️ AI didn't generate any fixes")
                
            else:
                print(f"❌ Workflow file not found: {workflow_path}")
                
        except Exception as e:
            print(f"❌ AI Brain failed to analyze: {e}")
            return False
    
    # Test 4: Check AI Brain's GitHub API access
    print(f"\n🔍 Test 4: AI Brain GitHub API Status")
    try:
        # Test API call
        headers = {
            'Authorization': f'token {os.environ["GITHUB_TOKEN"]}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        url = 'https://api.github.com/repos/c-trading-bo/trading-bot-c-/actions/runs'
        response = requests.get(url, headers=headers, params={'per_page': 1})
        
        if response.status_code == 200:
            print(f"✅ AI Brain has API access (rate limit: {response.headers.get('X-RateLimit-Remaining', 'unknown')})")
            return True
        else:
            print(f"❌ API access failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def check_ai_brain_auto_trigger():
    print(f"\n🔍 Checking AI Brain Auto-Trigger Status...")
    print("-" * 40)
    
    # Check if AI Brain workflow is configured correctly
    workflow_file = ".github/workflows/copilot_ai_mechanic.yml"
    if os.path.exists(workflow_file):
        with open(workflow_file, 'r') as f:
            content = f.read()
        
        # Check for trigger conditions
        if 'workflow_run:' in content and 'types:' in content and 'completed' in content:
            print("✅ AI Brain is configured to trigger on workflow failures")
        else:
            print("⚠️ AI Brain trigger configuration might be incomplete")
        
        if 'schedule:' in content and '*/30' in content:
            print("✅ AI Brain has scheduled monitoring every 30 minutes")
        else:
            print("⚠️ AI Brain scheduled monitoring not found")
    else:
        print("❌ AI Brain workflow file missing")

if __name__ == "__main__":
    print("🔬 AI BRAIN DETECTION TEST")
    print("=" * 60)
    
    success = test_ai_brain_detection()
    check_ai_brain_auto_trigger()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 AI BRAIN CAN SEE AND ANALYZE FAILURES!")
        print("💡 If it's not auto-fixing, there might be confidence threshold issues")
    else:
        print("⚠️ AI BRAIN HAS DETECTION ISSUES")
        print("💡 Need to debug the API access or workflow analysis")
