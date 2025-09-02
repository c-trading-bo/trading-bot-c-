#!/usr/bin/env python3
"""
Comprehensive workflow validator - validates all fixes
Tests YAML syntax, parameters, permissions, and more
"""

import os
import yaml
import json
import re
from datetime import datetime

def validate_workflows():
    """Check all workflows for issues and fixes"""
    
    print("="*70)
    print("🎯 COMPREHENSIVE WORKFLOW VALIDATION REPORT")
    print("="*70)
    
    workflow_dir = ".github/workflows"
    results = {
        'total_workflows': 0,
        'syntax_valid': 0,
        'permissions_fixed': 0,
        'checkout_updated': 0,
        'persist_credentials_added': 0,
        'parameter_issues': 0,
        'issues_found': [],
        'fixes_applied': []
    }
    
    for filename in os.listdir(workflow_dir):
        if not filename.endswith('.yml') and not filename.endswith('.yaml'):
            continue
        
        filepath = os.path.join(workflow_dir, filename)
        results['total_workflows'] += 1
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Test YAML syntax
            try:
                workflow_data = yaml.safe_load(content)
                results['syntax_valid'] += 1
                print(f"  ✅ {filename}: Valid YAML syntax")
            except yaml.YAMLError as e:
                results['issues_found'].append(f"{filename}: YAML syntax error - {e}")
                print(f"  ❌ {filename}: YAML syntax error")
                continue
            
            # Check for permissions
            if 'permissions:' in content:
                results['permissions_fixed'] += 1
                results['fixes_applied'].append(f"{filename}: Permissions added")
            else:
                results['issues_found'].append(f"{filename}: Missing permissions block")
            
            # Check for checkout version
            if 'actions/checkout@v4' in content:
                results['checkout_updated'] += 1
                results['fixes_applied'].append(f"{filename}: Checkout updated to v4")
            elif 'actions/checkout@v3' in content or 'actions/checkout@v2' in content:
                results['issues_found'].append(f"{filename}: Old checkout version")
            
            # Check for persist-credentials
            if 'persist-credentials: true' in content:
                results['persist_credentials_added'] += 1
                results['fixes_applied'].append(f"{filename}: Persist credentials added")
            elif 'git push' in content or 'git commit' in content:
                results['issues_found'].append(f"{filename}: Git operations without persist-credentials")
            
            # Check for parameter issues in ML workflows
            if 'train_cvar_ppo.py' in content:
                if '--cloud-mode' in content:
                    results['parameter_issues'] += 1
                    results['issues_found'].append(f"{filename}: Wrong parameter --cloud-mode")
                elif '--data' in content and '--save_dir' in content:
                    results['fixes_applied'].append(f"{filename}: Parameters corrected")
            
            # Check for API fallback in news/external API workflows
            if any(keyword in content.lower() for keyword in ['news', 'api', 'requests.get', 'feedparser']):
                if 'api_fallback' in content:
                    results['fixes_applied'].append(f"{filename}: API fallback added")
                else:
                    results['issues_found'].append(f"{filename}: Missing API fallback")
            
        except Exception as e:
            results['issues_found'].append(f"{filename}: File read error - {e}")
    
    # Print summary
    print(f"\n📊 VALIDATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total Workflows: {results['total_workflows']}")
    print(f"Valid YAML Syntax: {results['syntax_valid']}/{results['total_workflows']}")
    print(f"Permissions Fixed: {results['permissions_fixed']}/{results['total_workflows']}")
    print(f"Checkout Updated: {results['checkout_updated']}")
    print(f"Persist Credentials: {results['persist_credentials_added']}")
    print(f"Parameter Issues: {results['parameter_issues']}")
    
    print(f"\n✅ FIXES APPLIED ({len(results['fixes_applied'])}):")
    for fix in results['fixes_applied'][:10]:  # Show first 10
        print(f"  • {fix}")
    if len(results['fixes_applied']) > 10:
        print(f"  • ... and {len(results['fixes_applied']) - 10} more")
    
    if results['issues_found']:
        print(f"\n❌ REMAINING ISSUES ({len(results['issues_found'])}):")
        for issue in results['issues_found'][:10]:  # Show first 10
            print(f"  • {issue}")
        if len(results['issues_found']) > 10:
            print(f"  • ... and {len(results['issues_found']) - 10} more")
    else:
        print(f"\n🎉 NO REMAINING ISSUES FOUND!")
    
    # Calculate success rate
    success_rate = (results['syntax_valid'] / results['total_workflows']) * 100 if results['total_workflows'] > 0 else 0
    permissions_rate = (results['permissions_fixed'] / results['total_workflows']) * 100 if results['total_workflows'] > 0 else 0
    
    print(f"\n📈 SUCCESS METRICS:")
    print(f"  YAML Validity: {success_rate:.1f}%")
    print(f"  Permissions Coverage: {permissions_rate:.1f}%")
    print(f"  Overall Health: {'EXCELLENT' if success_rate > 95 and len(results['issues_found']) < 5 else 'GOOD' if success_rate > 90 else 'NEEDS_WORK'}")
    
    print("="*70)
    
    # Save detailed results
    with open('workflow_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return len(results['issues_found']) == 0

if __name__ == "__main__":
    success = validate_workflows()
    exit(0 if success else 1)
