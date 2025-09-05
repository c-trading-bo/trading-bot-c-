#!/usr/bin/env python3
"""
GitHub Actions Visibility Checker
Diagnoses why Cloud Bot Mechanic might not appear in Actions tab
"""

import os
import yaml
import json
from datetime import datetime

def check_workflow_visibility():
    print("🔍 GITHUB ACTIONS VISIBILITY DIAGNOSTIC")
    print("=" * 50)
    
    workflow_dir = ".github/workflows"
    cloud_mechanic_file = os.path.join(workflow_dir, "cloud_bot_mechanic.yml")
    
    # Check if file exists
    if not os.path.exists(cloud_mechanic_file):
        print("❌ cloud_bot_mechanic.yml NOT FOUND!")
        return
    
    print("✅ cloud_bot_mechanic.yml exists")
    
    # Check file size and content
    file_size = os.path.getsize(cloud_mechanic_file)
    print(f"📁 File size: {file_size} bytes")
    
    # Check YAML validity
    try:
        with open(cloud_mechanic_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for BOM
        if content.startswith('\ufeff'):
            print("⚠️  UTF-8 BOM detected (might cause issues)")
        else:
            print("✅ No UTF-8 BOM")
            
        # Parse YAML
        yaml_data = yaml.safe_load(content)
        print("✅ YAML syntax valid")
        
        # Check workflow structure
        if 'name' in yaml_data:
            print(f"✅ Workflow name: '{yaml_data['name']}'")
        else:
            print("❌ Missing 'name' field")
            
        if 'on' not in yaml_data:
            print("❌ Missing 'on' trigger section")
            return
            
        triggers = yaml_data['on']
        print(f"✅ Triggers configured: {list(triggers.keys())}")
        
        # Check schedule triggers
        if 'schedule' in triggers:
            schedules = triggers['schedule']
            print(f"✅ Schedule triggers: {len(schedules)}")
            for i, schedule in enumerate(schedules):
                print(f"   {i+1}. {schedule['cron']}")
        
        # Check workflow_run triggers
        if 'workflow_run' in triggers:
            workflow_run = triggers['workflow_run']
            if 'workflows' in workflow_run:
                workflows = workflow_run['workflows']
                print(f"✅ Workflow_run triggers: {len(workflows)} workflows")
                print("   Monitoring workflows:")
                for workflow in workflows[:5]:  # Show first 5
                    print(f"     - {workflow}")
                if len(workflows) > 5:
                    print(f"     ... and {len(workflows) - 5} more")
            else:
                print("❌ workflow_run missing 'workflows' list")
        
        # Check jobs
        if 'jobs' in yaml_data:
            jobs = yaml_data['jobs']
            print(f"✅ Jobs configured: {list(jobs.keys())}")
        else:
            print("❌ Missing 'jobs' section")
            
    except yaml.YAMLError as e:
        print(f"❌ YAML syntax error: {e}")
        return
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    print("\n🔍 POTENTIAL ISSUES CHECK:")
    
    # Check for common GitHub Actions issues
    issues = []
    
    # 1. File too large (GitHub has limits)
    if file_size > 1024 * 1024:  # 1MB
        issues.append("File might be too large for GitHub Actions")
    
    # 2. Too many workflow_run triggers
    if 'workflow_run' in triggers and len(triggers['workflow_run'].get('workflows', [])) > 50:
        issues.append("Too many workflow_run triggers (GitHub limit ~50)")
    
    # 3. Invalid cron expressions
    if 'schedule' in triggers:
        for schedule in triggers['schedule']:
            cron = schedule['cron']
            # Basic cron validation
            parts = cron.split()
            if len(parts) != 5:
                issues.append(f"Invalid cron expression: {cron}")
    
    # 4. Branch mismatch
    if 'workflow_run' in triggers:
        branches = triggers['workflow_run'].get('branches', ['main'])
        if 'main' not in branches and 'master' not in branches:
            issues.append("workflow_run not monitoring main/master branch")
    
    if issues:
        print("⚠️  POTENTIAL ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    else:
        print("✅ No obvious issues detected")
    
    print("\n🚀 TROUBLESHOOTING STEPS:")
    print("1. Check GitHub Actions tab manually")
    print("2. Try manual trigger via workflow_dispatch")
    print("3. Check repository settings -> Actions permissions")
    print("4. Verify branch is 'main' (not 'master')")
    print("5. Wait 5-10 minutes after push for GitHub to process")
    
    print(f"\n📊 SCAN COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    check_workflow_visibility()
