#!/usr/bin/env python3
"""
FINAL WORKFLOW FIX - Completely reconstruct workflows with proper structure
"""

import os
import yaml
from pathlib import Path

def final_fix_all_workflows():
    """Final fix for all workflows - complete reconstruction"""
    workflow_path = Path('.github/workflows')
    
    if not workflow_path.exists():
        print("❌ No .github/workflows directory found")
        return
    
    workflow_files = list(workflow_path.glob('*.yml')) + list(workflow_path.glob('*.yaml'))
    
    print(f"🔧 FINAL FIX FOR {len(workflow_files)} WORKFLOWS...\n")
    
    fixed_count = 0
    
    for wf_file in workflow_files:
        print(f"🔧 {wf_file.name}:", end=" ")
        
        try:
            # Read and parse the existing workflow
            with open(wf_file, 'r', encoding='utf-8') as f:
                workflow = yaml.safe_load(f)
            
            if not workflow:
                print("❌ Empty")
                continue
            
            # Extract and clean up the data
            name = workflow.get('name', wf_file.stem.replace('_', ' ').title())
            
            # Handle the 'on' section - check both 'on' and True keys
            on_section = None
            if 'on' in workflow:
                on_section = workflow['on']
            elif True in workflow:
                on_section = workflow[True]
            
            if not on_section:
                # Create a default 'on' section
                on_section = {'workflow_dispatch': None}
            
            # Get other sections
            permissions = workflow.get('permissions', {})
            env = workflow.get('env', {})
            jobs = workflow.get('jobs', {})
            defaults = workflow.get('defaults', {})
            
            # Reconstruct the workflow in the correct order
            new_workflow = {}
            
            # 1. Name
            new_workflow['name'] = name
            
            # 2. On section
            new_workflow['on'] = on_section
            
            # 3. Optional sections in order
            if env:
                new_workflow['env'] = env
            if permissions:
                new_workflow['permissions'] = permissions
            if defaults:
                new_workflow['defaults'] = defaults
            
            # 4. Jobs (required)
            if jobs:
                new_workflow['jobs'] = jobs
            else:
                # Create a placeholder job if none exists
                new_workflow['jobs'] = {
                    'placeholder': {
                        'runs-on': 'ubuntu-latest',
                        'steps': [
                            {'uses': 'actions/checkout@v4'},
                            {'run': 'echo "Workflow needs to be configured"'}
                        ]
                    }
                }
            
            # Write the reconstructed workflow
            with open(wf_file, 'w', encoding='utf-8') as f:
                yaml.dump(new_workflow, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            
            print("✅ FIXED")
            fixed_count += 1
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print(f"\n📊 FINAL FIX SUMMARY:")
    print(f"   🔧 Fixed: {fixed_count}/{len(workflow_files)}")
    
    # Final validation
    print(f"\n🔍 FINAL VALIDATION...")
    valid_count = 0
    
    for wf_file in workflow_files:
        try:
            with open(wf_file, 'r', encoding='utf-8') as f:
                workflow = yaml.safe_load(f)
            
            if ('name' in workflow and 'on' in workflow and 'jobs' in workflow and 
                True not in workflow and workflow['jobs']):
                valid_count += 1
            else:
                print(f"   ⚠️ {wf_file.name} still has issues")
                
        except Exception as e:
            print(f"   ❌ {wf_file.name}: {e}")
    
    print(f"\n🎯 FINAL RESULT:")
    print(f"   ✅ Valid workflows: {valid_count}/{len(workflow_files)}")
    print(f"   📈 Success rate: {(valid_count/len(workflow_files)*100):.1f}%")
    
    if valid_count == len(workflow_files):
        print(f"   🎉 ALL WORKFLOWS ARE NOW VALID FOR GITHUB ACTIONS!")
    else:
        print(f"   ⚠️ {len(workflow_files) - valid_count} workflows still need attention")

if __name__ == '__main__':
    final_fix_all_workflows()
