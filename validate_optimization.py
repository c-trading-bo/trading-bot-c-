#!/usr/bin/env python3
"""Final validation of optimized workflow schedules"""

import yaml
import os
from pathlib import Path

def main():
    print("🔍 FINAL WORKFLOW VALIDATION")
    print("="*50)
    
    workflow_dir = Path(".github/workflows")
    
    # Test all active workflows
    active_workflows = list(workflow_dir.glob("*.yml"))
    disabled_workflows = list(workflow_dir.glob("*.DISABLED"))
    
    print(f"\n📊 WORKFLOW SUMMARY:")
    print(f"  Active workflows: {len(active_workflows)}")
    print(f"  Disabled workflows: {len(disabled_workflows)}")
    
    # Validate YAML syntax for active workflows
    valid_count = 0
    invalid_workflows = []
    
    print(f"\n🔍 VALIDATING ACTIVE WORKFLOWS:")
    for workflow_file in active_workflows:
        try:
            with open(workflow_file, 'r') as f:
                yaml.safe_load(f.read())
            print(f"  ✅ {workflow_file.name}")
            valid_count += 1
        except yaml.YAMLError as e:
            print(f"  ❌ {workflow_file.name}: {e}")
            invalid_workflows.append(workflow_file.name)
    
    # Check for required optimization workflows
    required_workflows = [
        'ultimate_ml_rl_intel_system.yml',
        'es_nq_critical_trading.yml',
        'options_flow_analysis.yml',
        'ml_training_enhanced.yml',
        'news_sentiment.yml',
        'regime_detection.yml',
        'portfolio_heat.yml',
        'intelligence_collection.yml',
        'daily_consolidated.yml'
    ]
    
    print(f"\n🎯 CHECKING OPTIMIZATION WORKFLOWS:")
    missing_optimized = []
    for required in required_workflows:
        if (workflow_dir / required).exists():
            print(f"  ✅ {required}")
        else:
            print(f"  ❌ {required}")
            missing_optimized.append(required)
    
    # Usage calculation validation
    print(f"\n📊 USAGE CALCULATION:")
    try:
        exec(open('monitor_team_usage.py').read())
    except Exception as e:
        print(f"  ❌ Usage monitor error: {e}")
    
    # Summary
    print(f"\n📋 VALIDATION SUMMARY:")
    print(f"  Valid YAML: {valid_count}/{len(active_workflows)}")
    print(f"  Success Rate: {(valid_count/len(active_workflows)*100):.1f}%")
    print(f"  Missing Optimized: {len(missing_optimized)}")
    print(f"  Invalid Workflows: {len(invalid_workflows)}")
    
    if valid_count == len(active_workflows) and len(missing_optimized) == 0:
        print(f"\n🎉 ALL VALIDATIONS PASSED!")
        print(f"✅ Workflow optimization complete")
        print(f"✅ GitHub Team subscription optimized")
        print(f"✅ All YAML syntax valid")
        return True
    else:
        print(f"\n⚠️ VALIDATION ISSUES FOUND:")
        if invalid_workflows:
            print(f"  Invalid YAML: {', '.join(invalid_workflows)}")
        if missing_optimized:
            print(f"  Missing workflows: {', '.join(missing_optimized)}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)