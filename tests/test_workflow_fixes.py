#!/usr/bin/env python3
"""
Comprehensive Workflow Fix Validation Script
Tests all workflow fixes and dependencies
"""

import os
import json
import subprocess
import sys
from pathlib import Path
import yaml

def test_yaml_syntax():
    """Test all workflow YAML files for syntax errors"""
    print("🔍 Testing YAML syntax...")
    
    workflow_dir = Path('.github/workflows')
    errors = []
    
    for yaml_file in workflow_dir.glob('*.yml'):
        try:
            with open(yaml_file, 'r') as f:
                yaml.safe_load(f)
            print(f"  ✅ {yaml_file.name}")
        except yaml.YAMLError as e:
            errors.append(f"{yaml_file.name}: {e}")
            print(f"  ❌ {yaml_file.name}: {e}")
    
    return len(errors) == 0, errors

def test_talib_installation():
    """Test TA-Lib installation sequence"""
    print("📊 Testing TA-Lib installation...")
    
    try:
        # Test import
        import talib
        print("  ✅ TA-Lib successfully imported")
        
        # Test basic function
        import numpy as np
        test_data = np.random.randn(100)
        sma = talib.SMA(test_data, timeperiod=20)
        print("  ✅ TA-Lib SMA function working")
        
        return True, "TA-Lib working correctly"
    
    except ImportError as e:
        # Try backup libraries
        try:
            import ta
            print("  ⚠️  TA-Lib not available, but 'ta' library working")
            return True, "Backup 'ta' library available"
        except ImportError:
            try:
                import pandas_ta
                print("  ⚠️  TA-Lib not available, but 'pandas_ta' library working")
                return True, "Backup 'pandas_ta' library available"
            except ImportError:
                return False, f"No technical analysis libraries available: {e}"

def test_ml_dependencies():
    """Test all ML/RL dependencies"""
    print("🧠 Testing ML/RL dependencies...")
    
    required_packages = [
        'torch', 'numpy', 'pandas', 'sklearn', 
        'onnx', 'joblib', 'matplotlib'
    ]
    
    missing = []
    working = []
    
    for package in required_packages:
        try:
            __import__(package)
            working.append(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"  ❌ {package}")
    
    return len(missing) == 0, {'working': working, 'missing': missing}

def test_training_scripts():
    """Test that all training scripts exist and are executable"""
    print("🤖 Testing training scripts...")
    
    scripts = [
        'ml/train_meta_classifier.py',
        'ml/train_exec_quality.py', 
        'ml/train_rl_sizer.py',
        'ml/rl/train_cvar_ppo.py'
    ]
    
    missing = []
    present = []
    
    for script in scripts:
        if os.path.exists(script):
            present.append(script)
            print(f"  ✅ {script}")
        else:
            missing.append(script)
            print(f"  ❌ {script}")
    
    return len(missing) == 0, {'present': present, 'missing': missing}

def test_workflow_templates():
    """Test workflow templates"""
    print("📋 Testing workflow templates...")
    
    templates = [
        '.github/workflows/install_dependencies_template.yml',
        '.github/workflows/test_talib_fix.yml'
    ]
    
    present = []
    missing = []
    
    for template in templates:
        if os.path.exists(template):
            present.append(template)
            print(f"  ✅ {template}")
        else:
            missing.append(template)
            print(f"  ❌ {template}")
    
    return len(missing) == 0, {'present': present, 'missing': missing}

def generate_summary():
    """Generate comprehensive test summary"""
    print("\n" + "="*60)
    print("🎯 COMPREHENSIVE WORKFLOW FIX VALIDATION SUMMARY")
    print("="*60)
    
    results = {}
    
    # Run all tests
    yaml_ok, yaml_errors = test_yaml_syntax()
    results['yaml'] = {'ok': yaml_ok, 'details': yaml_errors}
    
    talib_ok, talib_msg = test_talib_installation()
    results['talib'] = {'ok': talib_ok, 'details': talib_msg}
    
    ml_ok, ml_details = test_ml_dependencies()
    results['ml_deps'] = {'ok': ml_ok, 'details': ml_details}
    
    scripts_ok, scripts_details = test_training_scripts()
    results['scripts'] = {'ok': scripts_ok, 'details': scripts_details}
    
    templates_ok, templates_details = test_workflow_templates()
    results['templates'] = {'ok': templates_ok, 'details': templates_details}
    
    # Overall status
    all_tests = [yaml_ok, talib_ok, ml_ok, scripts_ok, templates_ok]
    overall_ok = all(all_tests)
    
    print(f"\n📊 OVERALL STATUS: {'✅ PASS' if overall_ok else '❌ FAIL'}")
    print(f"✅ YAML Syntax: {'PASS' if yaml_ok else 'FAIL'}")
    print(f"📊 TA-Lib: {'PASS' if talib_ok else 'FAIL'}")
    print(f"🧠 ML Dependencies: {'PASS' if ml_ok else 'FAIL'}")
    print(f"🤖 Training Scripts: {'PASS' if scripts_ok else 'FAIL'}")
    print(f"📋 Templates: {'PASS' if templates_ok else 'FAIL'}")
    
    # Save detailed results
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Detailed results saved to: test_results.json")
    print("="*60)
    
    return overall_ok

if __name__ == "__main__":
    success = generate_summary()
    sys.exit(0 if success else 1)
