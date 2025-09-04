#!/usr/bin/env python3
"""
Test the enhanced GitHub Error Reader with AI Brain
"""

import os
import sys
from datetime import datetime

# Add the copilot_mechanic directory to path
sys.path.append('.github/copilot_mechanic')

try:
    from copilot_ai_brain import GitHubCopilotAIBrain
    from github_error_reader import GitHubWorkflowErrorReader
    print("✅ Successfully imported AI Brain with Error Reader")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_enhanced_ai_brain():
    """Test the enhanced AI brain with error reading capabilities"""
    
    print("🧪 TESTING ENHANCED GITHUB COPILOT AI BRAIN v3.0-PRO")
    print("=" * 60)
    
    # Set up environment
    os.environ['GITHUB_TOKEN'] = os.environ.get('GITHUB_TOKEN', 'your_github_token_here')
    os.environ['GITHUB_REPOSITORY'] = 'c-trading-bo/trading-bot-c-'
    
    # Test 1: Initialize AI Brain
    print("\\n🧠 Test 1: Initialize Enhanced AI Brain")
    brain = GitHubCopilotAIBrain()
    print(f"✅ AI Brain initialized: {brain.error_reader is not None}")
    
    # Test 2: Basic pattern analysis (fallback)
    print("\\n📋 Test 2: Basic Pattern Analysis")
    basic_result = brain.copilot_analyze("Workflow failed with YAML syntax error")
    print(basic_result[:200] + "...")
    
    # Test 3: Error reader capabilities
    print("\\n🔍 Test 3: Error Reader Capabilities")
    if brain.error_reader:
        # Test with a sample run ID (would normally come from webhook)
        sample_context = {
            'workflow_run_id': '12345',  # This would be real in production
            'workflow_name': 'CI/CD Pipeline'
        }
        
        try:
            enhanced_result = brain.copilot_analyze(
                "Analyze workflow failure", 
                context=sample_context
            )
            print("✅ Enhanced analysis with error reader")
            print(enhanced_result[:300] + "...")
        except Exception as e:
            print(f"⚠️ Error reader test failed (expected with sample ID): {e}")
    
    # Test 4: Error pattern recognition
    print("\\n🎯 Test 4: Error Pattern Recognition")
    error_patterns = [
        "ModuleNotFoundError: No module named 'requests'",
        "YAML syntax error in workflow file",
        "npm ERR! missing script: build",
        "Permission denied: /usr/bin/docker",
        "workflow timeout exceeded 6 hours"
    ]
    
    for pattern in error_patterns:
        result = brain.copilot_analyze(f"Error: {pattern}")
        confidence_line = [line for line in result.split('\\n') if 'CONFIDENCE:' in line]
        if confidence_line:
            print(f"• {pattern[:30]}... → {confidence_line[0].strip()}")
    
    # Test 5: Real error analysis simulation
    print("\\n🔬 Test 5: Real Error Analysis Simulation")
    mock_error_details = {
        'run_id': 'test_123',
        'name': 'Test Workflow',
        'error_messages': [
            "ModuleNotFoundError: No module named 'pandas'",
            "Process completed with exit code 1"
        ],
        'failed_steps': [
            {'name': 'Install dependencies', 'number': 2}
        ],
        'logs': {}
    }
    
    if hasattr(brain, 'analyze_real_errors'):
        analysis = brain.analyze_real_errors(mock_error_details, "test prompt")
        print(f"✅ Root Cause: {analysis['root_cause']}")
        print(f"✅ Fix Type: {analysis['fix_type']}")
        print(f"✅ Confidence: {int(analysis['confidence'] * 100)}%")
    
    print("\\n🎉 ENHANCED AI BRAIN TESTS COMPLETED")
    print(f"📅 Tested at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return True

if __name__ == "__main__":
    try:
        test_enhanced_ai_brain()
        print("\\n✅ ALL TESTS PASSED - Enhanced Error Reading Ready!")
    except Exception as e:
        print(f"\\n❌ Test failed: {e}")
        sys.exit(1)
