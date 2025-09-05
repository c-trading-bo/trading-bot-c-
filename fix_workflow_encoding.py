#!/usr/bin/env python3
"""
WORKFLOW ENCODING FIXER
Fixes character encoding issues in GitHub Actions workflow files
"""

import os
import glob
import yaml
from pathlib import Path

def fix_workflow_encoding():
    """Fix encoding issues in all workflow files"""
    
    workflows_dir = '.github/workflows'
    fixed_count = 0
    failed_files = []
    
    print("🔧 FIXING WORKFLOW ENCODING ISSUES...")
    print("=" * 50)
    
    for file_path in glob.glob(f'{workflows_dir}/*.yml'):
        filename = os.path.basename(file_path)
        
        try:
            # Try to read with different encodings
            content = None
            for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                print(f"❌ {filename} - Could not decode with any encoding")
                failed_files.append(filename)
                continue
            
            # Clean up problematic characters
            # Remove or replace common problematic characters
            content = content.replace('\u009d', '')  # Remove 0x9d character
            content = content.replace('\u008f', '')  # Remove 0x8f character  
            content = content.replace('\u0090', '')  # Remove 0x90 character
            content = content.replace('\u008d', '')  # Remove 0x8d character
            
            # Try to parse as YAML to validate
            try:
                yaml.safe_load(content)
                
                # Write back with clean UTF-8 encoding
                with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(content)
                
                print(f"✅ {filename} - Fixed encoding")
                fixed_count += 1
                
            except yaml.YAMLError as e:
                print(f"⚠️  {filename} - YAML syntax error after cleaning: {e}")
                failed_files.append(filename)
                
        except Exception as e:
            print(f"❌ {filename} - Error: {e}")
            failed_files.append(filename)
    
    print("\n" + "=" * 50)
    print(f"✅ FIXED: {fixed_count} workflows")
    print(f"❌ FAILED: {len(failed_files)} workflows")
    
    if failed_files:
        print(f"\n🚨 STILL FAILING:")
        for file in failed_files:
            print(f"  • {file}")
    
    return fixed_count, failed_files

if __name__ == "__main__":
    fixed_count, failed_files = fix_workflow_encoding()
    
    if fixed_count > 0:
        print(f"\n🎯 NEXT STEP: Commit and push {fixed_count} fixed workflow files")
    
    if failed_files:
        print(f"\n⚠️  MANUAL REVIEW NEEDED: {len(failed_files)} workflows still have issues")
