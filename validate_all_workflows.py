#!/usr/bin/env python3
"""
COMPREHENSIVE WORKFLOW VALIDATOR
Validates every single workflow file to ensure GitHub Actions compliance
"""

import os
import yaml
from pathlib import Path

def validate_and_fix_all_workflows():
    """Validate and fix every single workflow file"""
    workflow_path = Path('.github/workflows')
    
    if not workflow_path.exists():
        print("❌ No .github/workflows directory found")
        return
    
    workflow_files = list(workflow_path.glob('*.yml')) + list(workflow_path.glob('*.yaml'))
    
    print(f"🔍 VALIDATING {len(workflow_files)} WORKFLOWS...\n")
    
    valid_count = 0
    fixed_count = 0
    failed_count = 0
    
    for wf_file in workflow_files:
        print(f"📄 {wf_file.name}:")
        
        try:
            # Read the file
            with open(wf_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse YAML
            try:
                workflow = yaml.safe_load(content)
            except yaml.YAMLError as e:
                print(f"  ❌ YAML parsing error: {e}")
                failed_count += 1
                continue
            
            # Check if it's valid
            is_valid, issues = validate_workflow_structure(workflow, content)
            
            if is_valid:
                print(f"  ✅ VALID")
                valid_count += 1
            else:
                print(f"  🔧 FIXING ISSUES:")
                for issue in issues:
                    print(f"    - {issue}")
                
                # Attempt to fix
                if fix_workflow(wf_file, workflow, content, issues):
                    print(f"  ✅ FIXED")
                    fixed_count += 1
                else:
                    print(f"  ❌ FAILED TO FIX")
                    failed_count += 1
        
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failed_count += 1
    
    print(f"\n📊 VALIDATION SUMMARY:")
    print(f"   ✅ Valid: {valid_count}")
    print(f"   🔧 Fixed: {fixed_count}")
    print(f"   ❌ Failed: {failed_count}")
    print(f"   📈 Success Rate: {((valid_count + fixed_count) / len(workflow_files) * 100):.1f}%")

def validate_workflow_structure(workflow, content):
    """Validate a single workflow structure"""
    issues = []
    
    if not workflow:
        return False, ["Empty workflow file"]
    
    # Check required fields
    if 'name' not in workflow:
        issues.append("Missing 'name' field")
    
    if 'on' not in workflow:
        if "'on':" in content and content.index("'on':") > content.index("jobs:"):
            issues.append("'on' section is misplaced (should be after name, before jobs)")
        else:
            issues.append("Missing 'on' section")
    
    if 'jobs' not in workflow:
        issues.append("Missing 'jobs' section")
    elif not workflow['jobs']:
        issues.append("Empty jobs section")
    
    # Check for malformed triggers
    if True in workflow:
        issues.append("Invalid trigger structure (True key instead of 'on')")
    
    # Check for structure order (approximate)
    content_lines = content.split('\n')
    name_line = -1
    on_line = -1
    jobs_line = -1
    
    for i, line in enumerate(content_lines):
        if line.startswith('name:'):
            name_line = i
        elif line.strip() in ['on:', "'on':"]:
            on_line = i
        elif line.startswith('jobs:'):
            jobs_line = i
    
    if name_line > -1 and on_line > -1 and jobs_line > -1:
        if not (name_line < on_line < jobs_line):
            issues.append("Incorrect section order (should be: name -> on -> jobs)")
    
    return len(issues) == 0, issues

def fix_workflow(wf_file, workflow, content, issues):
    """Fix a workflow file"""
    try:
        lines = content.split('\n')
        
        # Extract sections
        name_section = []
        on_section = []
        other_sections = []
        
        current_section = 'other'
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            if line.startswith('name:'):
                name_section = [line]
                current_section = 'name'
            elif line.strip() in ['on:', "'on':"]:
                on_section = ['on:']
                current_section = 'on'
                # Collect the entire 'on' section
                i += 1
                while i < len(lines):
                    if lines[i].strip() and not lines[i].startswith(' ') and not lines[i].startswith('\t'):
                        i -= 1  # Back up one line
                        break
                    on_section.append(lines[i])
                    i += 1
                current_section = 'other'
            else:
                if current_section != 'on':
                    other_sections.append(line)
            
            i += 1
        
        # Handle True key issue
        if True in workflow:
            workflow['on'] = workflow.pop(True)
        
        # Reconstruct the file in proper order
        new_content = []
        
        # Add name section
        if name_section:
            new_content.extend(name_section)
        else:
            new_content.append(f"name: {wf_file.stem.replace('_', ' ').title()}")
        
        # Add on section
        if on_section and len(on_section) > 1:
            new_content.extend(on_section)
        elif 'on' in workflow:
            new_content.append('on:')
            if 'schedule' in workflow['on']:
                new_content.append('  schedule:')
                for schedule in workflow['on']['schedule']:
                    if isinstance(schedule, dict) and 'cron' in schedule:
                        new_content.append(f"  - cron: '{schedule['cron']}'")
            if 'workflow_dispatch' in workflow['on']:
                new_content.append('  workflow_dispatch:')
                wd = workflow['on']['workflow_dispatch']
                if wd and isinstance(wd, dict):
                    if 'inputs' in wd:
                        new_content.append('    inputs:')
                        for inp_name, inp_config in wd['inputs'].items():
                            new_content.append(f'      {inp_name}:')
                            for key, value in inp_config.items():
                                if isinstance(value, list):
                                    new_content.append(f'        {key}:')
                                    for item in value:
                                        new_content.append(f'        - {item}')
                                else:
                                    new_content.append(f'        {key}: {value}')
            if 'push' in workflow['on']:
                new_content.append('  push:')
                push_config = workflow['on']['push']
                if isinstance(push_config, dict) and 'branches' in push_config:
                    new_content.append('    branches:')
                    for branch in push_config['branches']:
                        new_content.append(f'    - {branch}')
            if 'pull_request' in workflow['on']:
                new_content.append('  pull_request:')
                pr_config = workflow['on']['pull_request']
                if isinstance(pr_config, dict) and 'branches' in pr_config:
                    new_content.append('    branches:')
                    for branch in pr_config['branches']:
                        new_content.append(f'    - {branch}')
        
        # Add other sections (permissions, env, jobs, etc.)
        for line in other_sections:
            if not line.strip().startswith("'on':") and 'on:' not in line:
                new_content.append(line)
        
        # Write the fixed file
        with open(wf_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_content))
        
        return True
        
    except Exception as e:
        print(f"    ❌ Fix error: {e}")
        return False

if __name__ == '__main__':
    validate_and_fix_all_workflows()
