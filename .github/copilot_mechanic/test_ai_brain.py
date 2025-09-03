#!/usr/bin/env python3
"""
Test and validate the Copilot Enterprise AI Brain system
"""

import os
import json
import sys
from pathlib import Path

# Add the copilot_mechanic directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from copilot_ai_brain import CopilotEnterpriseAIBrain
    print("✅ Successfully imported CopilotEnterpriseAIBrain")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_ai_brain_initialization():
    """Test AI Brain initialization"""
    
    print("\n🧪 Testing AI Brain Initialization...")
    
    try:
        # Set environment variables for testing
        os.environ['GITHUB_TOKEN'] = 'test-token'
        os.environ['GITHUB_REPOSITORY'] = 'c-trading-bo/trading-bot-c-'
        
        ai_brain = CopilotEnterpriseAIBrain()
        
        print(f"✅ AI Brain version: {ai_brain.version}")
        print(f"✅ Organization: {ai_brain.org}")
        print(f"✅ Repository: {ai_brain.repo}")
        print(f"✅ Confidence thresholds configured")
        
        return True
    
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return False

def test_workflow_analysis():
    """Test workflow analysis capabilities"""
    
    print("\n🧪 Testing Workflow Analysis...")
    
    try:
        os.environ['GITHUB_TOKEN'] = 'test-token'
        os.environ['GITHUB_REPOSITORY'] = 'c-trading-bo/trading-bot-c-'
        
        ai_brain = CopilotEnterpriseAIBrain()
        
        # Test analysis
        test_prompt = """
        WORKFLOW FAILURE ANALYSIS:
        Workflow: test-workflow
        Status: failure
        Error: syntax error in YAML
        """
        
        response = ai_brain.copilot_think(test_prompt)
        
        if "ROOT_CAUSE" in response and "CONFIDENCE" in response:
            print("✅ AI analysis format correct")
            print(f"✅ Response generated: {len(response)} characters")
            return True
        else:
            print("❌ AI analysis format incorrect")
            return False
    
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return False

def test_fix_parsing():
    """Test fix parsing capabilities"""
    
    print("\n🧪 Testing Fix Parsing...")
    
    try:
        os.environ['GITHUB_TOKEN'] = 'test-token'
        os.environ['GITHUB_REPOSITORY'] = 'c-trading-bo/trading-bot-c-'
        
        ai_brain = CopilotEnterpriseAIBrain()
        
        test_diagnosis = """
        ROOT_CAUSE: YAML syntax error
        FIX_TYPE: workflow_edit
        FIX_CODE:
        ```yaml
        name: Fixed Workflow
        on: push
        ```
        CONFIDENCE: 85
        PREVENTION: Use YAML validators
        """
        
        fix_data = ai_brain.parse_copilot_fix(test_diagnosis)
        
        if fix_data['confidence'] == 85 and 'workflow_edit' in fix_data['fix_type']:
            print("✅ Fix parsing works correctly")
            print(f"✅ Extracted confidence: {fix_data['confidence']}%")
            print(f"✅ Extracted fix type: {fix_data['fix_type']}")
            return True
        else:
            print("❌ Fix parsing failed")
            return False
    
    except Exception as e:
        print(f"❌ Parsing error: {e}")
        return False

def test_knowledge_system():
    """Test knowledge learning and storage"""
    
    print("\n🧪 Testing Knowledge System...")
    
    try:
        os.environ['GITHUB_TOKEN'] = 'test-token'
        os.environ['GITHUB_REPOSITORY'] = 'c-trading-bo/trading-bot-c-'
        
        ai_brain = CopilotEnterpriseAIBrain()
        
        # Test learning
        test_workflow = {'name': 'test-workflow', 'id': '123'}
        test_fix = {
            'root_cause': 'YAML syntax error',
            'fix_type': 'workflow_edit',
            'confidence': 85
        }
        
        ai_brain.learn_fix(test_workflow, test_fix)
        
        if ai_brain.knowledge_base['learned_fixes']:
            print("✅ Knowledge learning works")
            print(f"✅ Knowledge entries: {len(ai_brain.knowledge_base['learned_fixes'])}")
            return True
        else:
            print("❌ Knowledge learning failed")
            return False
    
    except Exception as e:
        print(f"❌ Knowledge error: {e}")
        return False

def test_file_structure():
    """Test file structure and permissions"""
    
    print("\n🧪 Testing File Structure...")
    
    required_dirs = [
        '.github/copilot_mechanic',
        '.github/copilot_mechanic/brain',
        '.github/copilot_mechanic/fixes',
        '.github/copilot_mechanic/knowledge'
    ]
    
    required_files = [
        '.github/copilot_mechanic/copilot_ai_brain.py',
        '.github/copilot_mechanic/config.yml',
        '.github/workflows/copilot_ai_mechanic.yml'
    ]
    
    all_good = True
    
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✅ Directory exists: {directory}")
        else:
            print(f"❌ Missing directory: {directory}")
            all_good = False
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ File exists: {file_path}")
        else:
            print(f"❌ Missing file: {file_path}")
            all_good = False
    
    return all_good

def main():
    """Run all tests"""
    
    print("="*60)
    print("🧠 COPILOT ENTERPRISE AI BRAIN - SYSTEM TEST")
    print("="*60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("AI Brain Initialization", test_ai_brain_initialization),
        ("Workflow Analysis", test_workflow_analysis),
        ("Fix Parsing", test_fix_parsing),
        ("Knowledge System", test_knowledge_system)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"Running: {test_name}")
        print(f"{'='*40}")
        
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Copilot Enterprise AI Brain is ready!")
        print("\nNext steps:")
        print("1. Push to GitHub to activate the AI mechanic")
        print("2. The AI will automatically monitor and fix workflow failures")
        print("3. Check the Actions tab to see the AI in action")
    else:
        print(f"\n⚠️ {total-passed} tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
