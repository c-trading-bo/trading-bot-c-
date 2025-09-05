#!/usr/bin/env python3
"""
Test script for Ultimate Cloud Mechanic integration
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from workflow_learner import WorkflowLearner
    from cloud_mechanic_core import CloudMechanicUltimate, CloudBotMechanic
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

async def test_workflow_learner():
    """Test the WorkflowLearner class"""
    print("\n🧪 Testing WorkflowLearner...")
    
    learner = WorkflowLearner()
    
    # Test pattern initialization
    assert len(learner.step_patterns) > 0, "Step patterns should be initialized"
    print(f"✅ Initialized with {len(learner.step_patterns)} step patterns")
    
    # Test YAML parsing
    test_yaml = """
name: Test Workflow
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Node
        uses: actions/setup-node@v3
        with:
          node-version: 18
      - run: npm install
      - run: npm test
    """
    
    learned = learner.parse_and_learn_workflow(test_yaml, "test.yml")
    assert learned['name'] == 'Test Workflow', "Should parse workflow name"
    assert 'test' in learned['jobs'], "Should parse job"
    assert len(learned['jobs']['test']['steps']) == 4, "Should parse all steps"
    print("✅ YAML parsing works correctly")
    
    # Test optimization generation
    optimizations = learned.get('optimizations', [])
    print(f"✅ Generated {len(optimizations)} optimizations")
    
    return True

async def test_ultimate_mechanic():
    """Test the CloudMechanicUltimate class"""
    print("\n🧪 Testing CloudMechanicUltimate...")
    
    # Set minimal environment for testing
    os.environ['GITHUB_REPOSITORY_OWNER'] = 'test-owner'
    os.environ['GITHUB_REPOSITORY'] = 'test-owner/test-repo'
    
    try:
        mechanic = CloudMechanicUltimate()
        print("✅ CloudMechanicUltimate initialized successfully")
        
        # Test metrics
        metrics = mechanic.get_ultimate_metrics()
        assert isinstance(metrics, dict), "Metrics should be a dictionary"
        assert 'workflows_learned' in metrics, "Should include learning metrics"
        print(f"✅ Metrics: {list(metrics.keys())}")
        
        # Test prediction (with empty repo info)
        predictions = await mechanic.predict_workflow_needs({'workflows': []})
        assert isinstance(predictions, dict), "Predictions should be a dictionary"
        print("✅ Workflow prediction works")
        
        return True
        
    except Exception as e:
        print(f"❌ Ultimate mechanic test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Testing Ultimate Cloud Mechanic Integration\n")
    
    # Test individual components
    learner_ok = await test_workflow_learner()
    ultimate_ok = await test_ultimate_mechanic()
    
    if learner_ok and ultimate_ok:
        print("\n✅ All tests passed! Ultimate Cloud Mechanic is ready.")
        return True
    else:
        print("\n❌ Some tests failed.")
        return False

if __name__ == "__main__":
    try:
        if sys.version_info >= (3, 7):
            result = asyncio.run(main())
        else:
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(main())
        
        sys.exit(0 if result else 1)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)