#!/usr/bin/env python3
"""
Local Workflow Diagnostics
Checks workflow files locally for GitHub Actions compatibility
"""

import os
import yaml
import json
from datetime import datetime

def check_workflow_files():
    """Check all workflow files for GitHub Actions compatibility"""
    
    workflow_dir = ".github/workflows"
    
    print("🔍 LOCAL WORKFLOW DIAGNOSTICS")
    print("=" * 50)
    print(f"Checking directory: {workflow_dir}")
    print(f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()
    
    if not os.path.exists(workflow_dir):
        print(f"❌ Workflow directory doesn't exist: {workflow_dir}")
        return
    
    workflow_files = [f for f in os.listdir(workflow_dir) if f.endswith('.yml') or f.endswith('.yaml')]
    
    print(f"📋 Found {len(workflow_files)} workflow files:")
    print()
    
    valid_workflows = 0
    invalid_workflows = 0
    ultimate_workflows = 0
    
    issues = []
    
    for filename in sorted(workflow_files):
        filepath = os.path.join(workflow_dir, filename)
        
        print(f"🔍 Checking: {filename}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for BOM
            if content.startswith('\ufeff'):
                issues.append(f"{filename}: Contains UTF-8 BOM (should be removed)")
            
            # Parse YAML
            try:
                workflow_data = yaml.safe_load(content)
                
                if not isinstance(workflow_data, dict):
                    issues.append(f"{filename}: Invalid YAML structure")
                    invalid_workflows += 1
                    continue
                
                # Check required fields
                required_fields = ['name', 'on', 'jobs']
                missing_fields = []
                
                for field in required_fields:
                    if field not in workflow_data:
                        missing_fields.append(field)
                
                if missing_fields:
                    issues.append(f"{filename}: Missing required fields: {missing_fields}")
                    invalid_workflows += 1
                    continue
                
                # Check if it's an Ultimate workflow
                name = workflow_data.get('name', '')
                if 'Ultimate' in name or 'ultimate' in filename.lower():
                    ultimate_workflows += 1
                    print(f"  🚀 ULTIMATE WORKFLOW: {name}")
                else:
                    print(f"  ✅ Regular workflow: {name}")
                
                # Check schedule syntax
                on_config = workflow_data.get('on', {})
                if isinstance(on_config, dict) and 'schedule' in on_config:
                    schedules = on_config['schedule']
                    if isinstance(schedules, list):
                        print(f"      📅 Has {len(schedules)} schedules")
                        for i, schedule in enumerate(schedules):
                            cron = schedule.get('cron', '')
                            if cron:
                                print(f"         {i+1}. {cron}")
                            else:
                                issues.append(f"{filename}: Schedule {i+1} missing cron expression")
                    else:
                        issues.append(f"{filename}: Schedule should be a list")
                else:
                    print(f"      ⚠️  No schedule found")
                
                # Check jobs structure
                jobs = workflow_data.get('jobs', {})
                if not isinstance(jobs, dict) or not jobs:
                    issues.append(f"{filename}: No jobs defined")
                    invalid_workflows += 1
                    continue
                
                print(f"      💼 Has {len(jobs)} jobs")
                
                # Check for permissions
                if 'permissions' not in workflow_data:
                    issues.append(f"{filename}: Missing permissions section (recommended)")
                
                valid_workflows += 1
                
            except yaml.YAMLError as e:
                issues.append(f"{filename}: YAML parsing error: {e}")
                invalid_workflows += 1
                
        except Exception as e:
            issues.append(f"{filename}: File reading error: {e}")
            invalid_workflows += 1
        
        print()
    
    # Summary
    print("=" * 50)
    print("📊 SUMMARY:")
    print(f"✅ Valid workflows: {valid_workflows}")
    print(f"❌ Invalid workflows: {invalid_workflows}")
    print(f"🚀 Ultimate workflows: {ultimate_workflows}")
    print(f"⚠️  Issues found: {len(issues)}")
    
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"  • {issue}")
    
    print("\n🔍 GITHUB ACTIONS CHECKLIST:")
    print("✅ Repository has .github/workflows/ directory")
    print(f"✅ Found {len(workflow_files)} .yml files")
    
    if valid_workflows > 0:
        print("✅ At least one valid workflow found")
    else:
        print("❌ No valid workflows found")
    
    if ultimate_workflows > 0:
        print(f"✅ Found {ultimate_workflows} Ultimate workflows")
    
    print("\n🚨 POSSIBLE REASONS WORKFLOWS AREN'T SHOWING:")
    print("1. Repository is private (most likely)")
    print("2. GitHub Actions is disabled in repository settings") 
    print("3. Workflows have syntax errors (check issues above)")
    print("4. No push to main branch yet (workflows need to be on main)")
    print("5. Workflows are scheduled but haven't triggered yet")
    
    print("\n💡 TROUBLESHOOTING STEPS:")
    print("1. Check GitHub repository settings > Actions")
    print("2. Ensure repository is public OR you're logged in")
    print("3. Check GitHub Actions tab directly in web interface")
    print("4. Push a small change to trigger workflow_dispatch")
    print("5. Check repository's Actions permissions")

if __name__ == "__main__":
    check_workflow_files()
