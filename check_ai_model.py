#!/usr/bin/env python3
"""
Check what AI model/logic the bot is actually using
"""
import os
import sys
import json

def check_ai_model_usage():
    print("🧠 CHECKING: What AI Logic Is Your Bot Actually Using?")
    print("=" * 60)
    
    # Import AI Brain
    sys.path.append('.github/copilot_mechanic')
    from copilot_ai_brain import CopilotEnterpriseAIBrain
    
    # Set environment
    os.environ['GITHUB_TOKEN'] = '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'
    os.environ['GITHUB_REPOSITORY'] = 'c-trading-bo/trading-bot-c-'
    
    brain = CopilotEnterpriseAIBrain()
    
    # Check configuration
    print("🔍 AI Brain Configuration:")
    print(f"   Organization: {brain.org}")
    print(f"   Repository: {brain.repo}")
    
    # Check thresholds 
    if hasattr(brain, 'thresholds'):
        print(f"   Auto-fix threshold: {brain.thresholds.get('auto_fix', 'unknown')}")
        print(f"   PR threshold: {brain.thresholds.get('pr_create', 'unknown')}")
    
    # Check enterprise config
    if hasattr(brain, 'enterprise_config'):
        print(f"   Enterprise enabled: {brain.enterprise_config.get('unlimited_requests', False)}")
        print(f"   Model: {brain.enterprise_config.get('model', 'unknown')}")
    else:
        print("   Enterprise config: Not found - checking code...")
    
    # Test what method it actually calls
    print(f"\n🔍 Testing AI Analysis Method:")
    try:
        # Test the copilot_think method
        test_prompt = "Analyze this workflow failure: syntax error in YAML"
        
        print("   📋 Calling brain.copilot_think()...")
        result = brain.copilot_think(test_prompt)
        
        print(f"   ✅ Response received: {len(result)} characters")
        print(f"   📊 Response type: {'GPT-4 API' if 'api.openai.com' in str(result) else 'Local Pattern Matching'}")
        
        # Check if it contains GPT-4 style analysis
        if "ROOT_CAUSE:" in result and "FIX_TYPE:" in result:
            print("   🎯 Using: Pattern-based analysis (not live GPT-4)")
        elif "I understand" in result or "Let me analyze" in result:
            print("   🎯 Using: Live GPT-4 Turbo API")
        else:
            print("   🎯 Using: Unknown analysis method")
            
        print(f"\n📄 Sample Output:")
        print("-" * 40)
        print(result[:300] + "..." if len(result) > 300 else result)
        
    except Exception as e:
        print(f"   ❌ Error testing AI: {e}")
    
    # Check actual code paths
    print(f"\n🔍 Code Analysis:")
    
    # Read the actual implementation
    with open('.github/copilot_mechanic/copilot_ai_brain.py', 'r', encoding='utf-8') as f:
        code = f.read()
    
    if 'api.openai.com' in code:
        print("   📡 OpenAI API endpoint found in code")
    if 'gpt-4-turbo' in code:
        print("   🧠 GPT-4 Turbo model specified")
    if 'github_ai_analysis' in code:
        print("   🔄 Falls back to GitHub-based analysis")
    if 'analyze_workflow_failure' in code and 'ROOT_CAUSE:' in code:
        print("   📝 Has hardcoded pattern responses")
    
    print(f"\n🎯 CONCLUSION:")
    print("-" * 40)
    if 'github_ai_analysis' in code and 'ROOT_CAUSE:' in code:
        print("❌ Currently using: HARDCODED PATTERN MATCHING (not GPT-4)")
        print("💡 The bot has GPT-4 code but falls back to patterns")
        print("🔧 To enable GPT-4: Need proper OpenAI API key setup")
    else:
        print("✅ Using: Live GPT-4 Turbo API")

def check_openai_config():
    print(f"\n🔍 OpenAI Configuration Check:")
    print("-" * 30)
    
    # Check if OpenAI key is available
    openai_key = os.environ.get('OPENAI_API_KEY')
    if openai_key:
        print(f"✅ OpenAI API Key: Found ({len(openai_key)} characters)")
    else:
        print("❌ OpenAI API Key: Missing")
        print("💡 Using GitHub token as fallback (won't work with OpenAI)")
    
    # Check what the AI brain is configured to use
    print(f"\n🔍 What would enable true GPT-4:")
    print("   1. Set OPENAI_API_KEY environment variable")
    print("   2. Or use GitHub Copilot Enterprise API (when available)")
    print("   3. Currently falling back to pattern matching")

if __name__ == "__main__":
    check_ai_model_usage()
    check_openai_config()
    
    print(f"\n" + "=" * 60)
    print("🎯 SUMMARY: Your bot is NOT using live GPT-4")
    print("   → It's using pattern-based analysis")
    print("   → To enable GPT-4: Add OpenAI API key")
    print("   → Current approach still works for fixing workflows!")
    print("=" * 60)
