#!/usr/bin/env python3
"""
🔍 GitHub Actions Workflow Health Check
Analyzes workflow files for common issues that cause Actions failures
"""

import os
import yaml
import json
from pathlib import Path

def check_workflow_health():
    """Check all workflow files for common issues"""
    
    workflow_dir = Path('.github/workflows')
    issues = []
    warnings = []
    successes = []
    
    if not workflow_dir.exists():
        return {"error": "No .github/workflows directory found"}
    
    print("🔍 WORKFLOW HEALTH CHECK STARTING...")
    print("=" * 50)
    
    for workflow_file in workflow_dir.glob('*.yml'):
        print(f"\n📄 Checking: {workflow_file.name}")
        
        try:
            with open(workflow_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Basic YAML syntax check
            try:
                workflow_data = yaml.safe_load(content)
                print(f"  ✅ YAML syntax: Valid")
                successes.append(f"{workflow_file.name}: YAML syntax valid")
            except yaml.YAMLError as e:
                issues.append(f"{workflow_file.name}: YAML syntax error - {e}")
                print(f"  ❌ YAML syntax: ERROR - {e}")
                continue
            
            # Check for required fields
            if 'name' not in workflow_data:
                warnings.append(f"{workflow_file.name}: Missing 'name' field")
                print(f"  ⚠️  Missing 'name' field")
            else:
                print(f"  ✅ Name field: Present")
            
            if 'on' not in workflow_data:
                issues.append(f"{workflow_file.name}: Missing 'on' trigger")
                print(f"  ❌ Missing 'on' trigger")
            else:
                print(f"  ✅ Trigger: Present")
            
            if 'jobs' not in workflow_data:
                issues.append(f"{workflow_file.name}: Missing 'jobs' section")
                print(f"  ❌ Missing 'jobs' section")
            else:
                print(f"  ✅ Jobs: Present ({len(workflow_data['jobs'])} jobs)")
            
            # Check for secrets usage
            secret_references = content.count('secrets.')
            if secret_references > 0:
                warnings.append(f"{workflow_file.name}: Uses {secret_references} secrets (ensure they're configured)")
                print(f"  ⚠️  Uses {secret_references} secrets (check GitHub repo settings)")
            
            # Check for optimization markers
            if '⚡' in content or 'OPTIMIZED' in content:
                print(f"  🚀 Optimized: Yes")
                successes.append(f"{workflow_file.name}: Contains optimization markers")
            else:
                warnings.append(f"{workflow_file.name}: No optimization markers found")
                print(f"  📈 Optimized: No markers found")
                
        except Exception as e:
            issues.append(f"{workflow_file.name}: File read error - {e}")
            print(f"  ❌ File read: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print("📊 HEALTH CHECK SUMMARY")
    print("=" * 50)
    
    print(f"\n✅ SUCCESSES ({len(successes)}):")
    for success in successes:
        print(f"  • {success}")
    
    print(f"\n⚠️  WARNINGS ({len(warnings)}):")
    for warning in warnings:
        print(f"  • {warning}")
    
    print(f"\n❌ CRITICAL ISSUES ({len(issues)}):")
    for issue in issues:
        print(f"  • {issue}")
    
    # Overall health score
    total_checks = len(successes) + len(warnings) + len(issues)
    health_score = (len(successes) / total_checks * 100) if total_checks > 0 else 0
    
    print(f"\n🎯 OVERALL HEALTH SCORE: {health_score:.1f}%")
    
    if len(issues) == 0:
        print("✅ NO CRITICAL ISSUES - Workflows should run!")
    else:
        print(f"🚨 {len(issues)} CRITICAL ISSUES NEED FIXING")
    
    if len(warnings) > 0:
        print(f"📋 {len(warnings)} warnings (may cause Actions failures)")
    
    return {
        "health_score": health_score,
        "successes": successes,
        "warnings": warnings,
        "issues": issues
    }

if __name__ == "__main__":
    result = check_workflow_health()
    
    # Save results
    with open('data/workflow_health_check.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n💾 Results saved to: data/workflow_health_check.json")
