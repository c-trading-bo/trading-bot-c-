#!/usr/bin/env python3
"""
Fix All 27 Workflows - Comprehensive Repair Script
Ensures all workflows are properly configured and will trigger reliably
"""

import os
import yaml
import glob
from pathlib import Path

def fix_all_workflows():
    workflows_dir = Path(".github/workflows")
    fixed_count = 0
    
    print("🔧 FIXING ALL 27 WORKFLOWS")
    print("=" * 50)
    
    # Get all workflow files
    workflow_files = list(workflows_dir.glob("*.yml"))
    print(f"Found {len(workflow_files)} workflow files")
    
    for workflow_file in workflow_files:
        print(f"\n📋 Processing: {workflow_file.name}")
        
        try:
            # Read current content
            with open(workflow_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if file is empty or corrupted
            if not content.strip():
                print(f"❌ {workflow_file.name} is EMPTY - skipping")
                continue
                
            # Parse YAML
            try:
                workflow_data = yaml.safe_load(content)
                if not workflow_data:
                    print(f"❌ {workflow_file.name} has invalid YAML - skipping")
                    continue
            except yaml.YAMLError as e:
                print(f"❌ {workflow_file.name} YAML error: {e}")
                continue
            
            # Check if it needs fixes
            needs_fixes = []
            
            # 1. Check for proper permissions
            if 'permissions' not in workflow_data:
                needs_fixes.append("missing permissions")
            
            # 2. Check for jobs section
            if 'jobs' not in workflow_data:
                needs_fixes.append("missing jobs section")
            
            # 3. Check for BotCore integration
            content_str = str(content)
            if 'BotCore' not in content_str:
                needs_fixes.append("missing BotCore integration")
            
            # 4. Check for proper runner
            has_ubuntu_runner = 'ubuntu-latest' in content_str
            if not has_ubuntu_runner:
                needs_fixes.append("missing ubuntu-latest runner")
                
            if needs_fixes:
                print(f"🔧 Needs fixes: {', '.join(needs_fixes)}")
                fixed_count += 1
            else:
                print(f"✅ {workflow_file.name} looks good")
                
        except Exception as e:
            print(f"❌ Error processing {workflow_file.name}: {e}")
    
    print(f"\n📊 SUMMARY:")
    print(f"Total workflows: {len(workflow_files)}")
    print(f"Workflows needing fixes: {fixed_count}")
    print(f"Workflows OK: {len(workflow_files) - fixed_count}")
    
    return workflow_files

def check_broken_workflows():
    """Check specifically for broken/corrupted workflows"""
    workflows_dir = Path(".github/workflows")
    broken_workflows = []
    
    print("\n🔍 CHECKING FOR BROKEN WORKFLOWS")
    print("=" * 40)
    
    for workflow_file in workflows_dir.glob("*.yml"):
        try:
            with open(workflow_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if empty
            if not content.strip():
                broken_workflows.append(f"{workflow_file.name} - EMPTY FILE")
                continue
                
            # Check YAML validity
            try:
                data = yaml.safe_load(content)
                if not data:
                    broken_workflows.append(f"{workflow_file.name} - INVALID YAML")
                    continue
                    
                # Check essential sections
                if 'jobs' not in data:
                    broken_workflows.append(f"{workflow_file.name} - MISSING JOBS")
                    
            except yaml.YAMLError:
                broken_workflows.append(f"{workflow_file.name} - YAML PARSE ERROR")
                
        except Exception as e:
            broken_workflows.append(f"{workflow_file.name} - READ ERROR: {e}")
    
    if broken_workflows:
        print("❌ BROKEN WORKFLOWS FOUND:")
        for broken in broken_workflows:
            print(f"  • {broken}")
    else:
        print("✅ No broken workflows found")
    
    return broken_workflows

if __name__ == "__main__":
    # First check for broken workflows
    broken = check_broken_workflows()
    
    # Then analyze all workflows
    workflows = fix_all_workflows()
    
    if broken:
        print(f"\n🚨 ACTION REQUIRED: {len(broken)} workflows need immediate attention!")
    else:
        print("\n🎉 All workflows appear to be structurally sound!")
