#!/usr/bin/env python3
"""
Fix workflow YAML files to properly quote the 'on' key to avoid YAML parsing issues
"""

import os
import re
from pathlib import Path

def fix_on_key_in_workflow(file_path):
    """Fix the 'on:' key in a workflow file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace 'on:' with '"on":' to avoid YAML boolean parsing
        # But only at the beginning of lines (not in comments)
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # If line starts with 'on:' (possibly with whitespace), quote it
            if re.match(r'^(\s*)on\s*:', line):
                fixed_line = re.sub(r'^(\s*)on(\s*):', r'\1"on"\2:', line)
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
        
        fixed_content = '\n'.join(fixed_lines)
        
        # Write the fixed content back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        return True, f"Fixed 'on' key in {file_path}"
        
    except Exception as e:
        return False, f"Error fixing {file_path}: {e}"

def main():
    """Fix all workflow files"""
    print("🔧 FIXING 'on' KEYS IN ALL WORKFLOW YAML FILES")
    print("=" * 50)
    
    workflows_dir = ".github/workflows"
    
    if not os.path.exists(workflows_dir):
        print(f"❌ ERROR: Workflows directory not found: {workflows_dir}")
        return
    
    workflow_files = [f for f in os.listdir(workflows_dir) if f.endswith('.yml')]
    workflow_files.sort()
    
    print(f"Found {len(workflow_files)} workflow files to fix\n")
    
    fixed_count = 0
    error_count = 0
    
    for workflow_file in workflow_files:
        file_path = os.path.join(workflows_dir, workflow_file)
        
        success, message = fix_on_key_in_workflow(file_path)
        
        if success:
            print(f"✅ {workflow_file}")
            fixed_count += 1
        else:
            print(f"❌ {workflow_file}: {message}")
            error_count += 1
    
    print(f"\n📊 SUMMARY:")
    print(f"✅ Fixed: {fixed_count}")
    print(f"❌ Errors: {error_count}")
    
    if fixed_count > 0:
        print("\n🎉 'on' keys have been fixed!")
        print("🔄 Please run verification again to check the results.")

if __name__ == "__main__":
    main()