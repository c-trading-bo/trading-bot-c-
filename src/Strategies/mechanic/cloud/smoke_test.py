#!/usr/bin/env python3
"""
SMOKE TEST - Test all cloud mechanic modules
"""

import sys
import os
from pathlib import Path

# Add the cloud mechanic path
cloud_path = Path(__file__).parent
sys.path.insert(0, str(cloud_path))

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing module imports...")
    
    try:
        from cloud_mechanic_core import CloudBotMechanic
        print("  ✅ cloud_mechanic_core")
    except Exception as e:
        print(f"  ❌ cloud_mechanic_core: {e}")
        return False
    
    try:
        from workflow_analyzer import WorkflowAnalyzer
        print("  ✅ workflow_analyzer")
    except Exception as e:
        print(f"  ❌ workflow_analyzer: {e}")
        return False
    
    try:
        from workflow_optimizer import WorkflowOptimizer
        print("  ✅ workflow_optimizer")
    except Exception as e:
        print(f"  ❌ workflow_optimizer: {e}")
        return False
    
    try:
        from repair_system import RepairSystem
        print("  ✅ repair_system")
    except Exception as e:
        print(f"  ❌ repair_system: {e}")
        return False
    
    try:
        from report_generator import ReportGenerator
        print("  ✅ report_generator")
    except Exception as e:
        print(f"  ❌ report_generator: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality"""
    print("\n🔧 Testing basic functionality...")
    
    try:
        from cloud_mechanic_core import CloudBotMechanic
        
        # Create instance
        mechanic = CloudBotMechanic()
        print(f"  ✅ CloudBotMechanic created (v{mechanic.version})")
        
        # Test file operations
        test_data = {"test": "data", "timestamp": "2024-01-01"}
        test_path = cloud_path / "test_output.json"
        
        mechanic.save_json(test_path, test_data)
        loaded_data = mechanic.load_json(test_path, {})
        
        if loaded_data.get("test") == "data":
            print("  ✅ JSON save/load works")
            test_path.unlink()  # Clean up
        else:
            print("  ❌ JSON save/load failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        return False

def test_workflow_analysis():
    """Test workflow analysis on existing workflows"""
    print("\n📊 Testing workflow analysis...")
    
    try:
        from workflow_analyzer import WorkflowAnalyzer
        
        analyzer = WorkflowAnalyzer()
        result = analyzer.analyze_workflows()
        
        print(f"  ✅ Workflow analysis completed")
        print(f"     Workflows found: {len(result.get('workflows', {}))}")
        print(f"     Total jobs: {result.get('summary', {}).get('total_jobs', 0)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Workflow analysis failed: {e}")
        return False

def main():
    """Run smoke test"""
    print("🌩️ CLOUD MECHANIC SMOKE TEST\n")
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_workflow_analysis
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n📊 SMOKE TEST RESULTS")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    
    if failed == 0:
        print("   🎉 ALL TESTS PASSED - Cloud Mechanic is ready!")
        return 0
    else:
        print("   ⚠️ Some tests failed - check the output above")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
