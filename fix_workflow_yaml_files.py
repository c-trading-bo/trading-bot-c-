#!/usr/bin/env python3
"""
Fix all workflow YAML files by removing line numbers and ensuring proper format
"""

import os
import re
import shutil
from pathlib import Path

def clean_workflow_file(file_path):
    """Clean a workflow file by removing line numbers and fixing formatting"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create backup
        backup_path = f"{file_path}.backup"
        shutil.copy2(file_path, backup_path)
        
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove line numbers from the beginning (pattern like "1.name:" or "  2.on:")
            # Look for patterns like "number." at the start of lines
            cleaned_line = re.sub(r'^\s*\d+\.', '', line)
            cleaned_lines.append(cleaned_line)
        
        cleaned_content = '\n'.join(cleaned_lines)
        
        # Write the cleaned content back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        return True, f"Cleaned {file_path}"
        
    except Exception as e:
        return False, f"Error cleaning {file_path}: {e}"

def main():
    """Clean all workflow files"""
    print("🧹 FIXING ALL WORKFLOW YAML FILES")
    print("=" * 50)
    
    workflows_dir = ".github/workflows"
    
    if not os.path.exists(workflows_dir):
        print(f"❌ ERROR: Workflows directory not found: {workflows_dir}")
        return
    
    workflow_files = [f for f in os.listdir(workflows_dir) if f.endswith('.yml')]
    workflow_files.sort()
    
    print(f"Found {len(workflow_files)} workflow files to clean\n")
    
    cleaned_count = 0
    error_count = 0
    
    for workflow_file in workflow_files:
        file_path = os.path.join(workflows_dir, workflow_file)
        
        success, message = clean_workflow_file(file_path)
        
        if success:
            print(f"✅ {workflow_file}")
            cleaned_count += 1
        else:
            print(f"❌ {workflow_file}: {message}")
            error_count += 1
    
    print(f"\n📊 SUMMARY:")
    print(f"✅ Cleaned: {cleaned_count}")
    print(f"❌ Errors: {error_count}")
    print(f"📁 Backups created with .backup extension")
    
    if cleaned_count > 0:
        print("\n🎉 Workflow files have been cleaned!")
        print("🔄 Please run verification again to check the results.")

if __name__ == "__main__":
    main()