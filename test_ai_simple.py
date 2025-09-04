#!/usr/bin/env python3
"""
Test AI Brain Detection of GitHub Failures
"""
import os
import sys
import requests
import json

# Set environment
os.environ['GITHUB_TOKEN'] = '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'
os.environ['GITHUB_REPOSITORY'] = 'c-trading-bo/trading-bot-c-'

def test_ai_detection():
    print("🧠 Testing AI Brain Failure Detection...")
    print("=" * 50)
    
    # Import AI Brain
    sys.path.append('.github/copilot_mechanic')
    from copilot_ai_brain import CopilotEnterpriseAIBrain
    
    brain = CopilotEnterpriseAIBrain()
    
    # Test 1: Get recent failures
    print("🔍 Test 1: Can AI detect recent failures?")
    try:
        failures = brain.get_recent_failures("news_sentiment", limit=3)
        print(f"✅ Found {len(failures)} recent failures in news_sentiment")
        
        for failure in failures:
            print(f"   ❌ ID: {failure.get('id', 'unknown')}")
            print(f"      Status: {failure.get('conclusion', 'unknown')}")
            print(f"      Time: {failure.get('created_at', 'unknown')}")
    except Exception as e:
        print(f"❌ Failed to get recent failures: {e}")
    
    # Test 2: Analyze workflow YAML
    print(f"\n🔍 Test 2: Can AI analyze workflow YAML?")
    try:
        yaml_content = brain.get_workflow_yaml(".github/workflows/news_sentiment.yml")
        print(f"✅ Loaded YAML: {len(yaml_content)} characters")
        
        # Analyze the workflow
        analysis = brain.analyze_workflow_failure(yaml_content, {"workflow": "news_sentiment"})
        print(f"✅ Generated analysis: {len(analysis)} characters")
        print(f"📊 Analysis snippet: {analysis[:150]}...")
        
    except Exception as e:
        print(f"❌ Failed to analyze YAML: {e}")
    
    # Test 3: Test diagnose and fix
    print(f"\n🔍 Test 3: Can AI diagnose and suggest fixes?")
    try:
        # Simulate a failed workflow run
        mock_workflow_run = {
            "id": "17449041976",
            "name": "news_sentiment.yml",
            "conclusion": "failure",
            "html_url": "https://github.com/c-trading-bo/trading-bot-c-/actions/runs/17449041976"
        }
        
        diagnosis = brain.diagnose_and_fix_workflow(mock_workflow_run)
        print(f"✅ AI generated diagnosis with {len(diagnosis)} fields")
        
        if diagnosis.get('confidence', 0) > 0:
            print(f"🎯 Confidence: {diagnosis.get('confidence', 0)}%")
            print(f"🔧 Fix type: {diagnosis.get('fix_type', 'unknown')}")
            print(f"💡 Root cause: {diagnosis.get('root_cause', 'unknown')}")
            
            # Check if it would auto-fix
            if diagnosis.get('confidence', 0) >= 85:
                print("🚀 AI WOULD AUTO-FIX THIS (85%+ confidence)")
            elif diagnosis.get('confidence', 0) >= 60:
                print("📝 AI WOULD CREATE PR (60%+ confidence)")
            else:
                print("⚠️ Low confidence - AI would just log the issue")
        else:
            print("⚠️ AI couldn't determine confidence level")
            
    except Exception as e:
        print(f"❌ Failed to diagnose: {e}")
    
    # Test 4: Direct API test
    print(f"\n🔍 Test 4: Direct GitHub API access")
    try:
        headers = {
            'Authorization': f'token {os.environ["GITHUB_TOKEN"]}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        # Get workflow runs
        url = 'https://api.github.com/repos/c-trading-bo/trading-bot-c-/actions/runs'
        response = requests.get(url, headers=headers, params={'per_page': 5})
        
        if response.status_code == 200:
            data = response.json()
            runs = data.get('workflow_runs', [])
            failed_runs = [r for r in runs if r.get('conclusion') == 'failure']
            
            print(f"✅ API Access OK - Found {len(failed_runs)} recent failures")
            print(f"📊 Rate limit remaining: {response.headers.get('X-RateLimit-Remaining', 'unknown')}")
            
            # Show what AI should be seeing
            for run in failed_runs[:3]:
                print(f"   ❌ {run.get('name', 'Unknown')}")
                print(f"      ID: {run.get('id')}")
                print(f"      Conclusion: {run.get('conclusion')}")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ API test failed: {e}")

def check_auto_trigger_config():
    print(f"\n🔍 Checking Auto-Trigger Configuration...")
    print("-" * 40)
    
    try:
        with open(".github/workflows/copilot_ai_mechanic.yml", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check trigger conditions
        triggers_found = []
        if 'workflow_run:' in content:
            triggers_found.append("✅ Workflow failure trigger")
        if 'schedule:' in content and '*/30' in content:
            triggers_found.append("✅ 30-minute schedule trigger")
        if 'workflow_dispatch:' in content:
            triggers_found.append("✅ Manual trigger")
        
        print("AI Brain triggers configured:")
        for trigger in triggers_found:
            print(f"   {trigger}")
            
        if len(triggers_found) >= 2:
            print("\n🎉 AI Brain is properly configured to monitor failures!")
        else:
            print("\n⚠️ AI Brain trigger configuration needs attention")
            
    except Exception as e:
        print(f"❌ Failed to check config: {e}")

if __name__ == "__main__":
    print("🔬 AI BRAIN GITHUB DETECTION TEST")
    print("=" * 50)
    
    test_ai_detection()
    check_auto_trigger_config()
    
    print("\n" + "=" * 50)
    print("💡 KEY FINDINGS:")
    print("   - If AI can see failures but not fixing them:")
    print("     → Check confidence thresholds (85% for auto-fix)")
    print("   - If API access works but AI doesn't trigger:")
    print("     → Check workflow_run trigger configuration")
    print("   - Your screenshot shows failures that AI should detect!")
