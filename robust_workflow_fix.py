#!/usr/bin/env python3
import os
import re
import yaml
from pathlib import Path

def fix_workflows():
    """
    Robust workflow fixer that handles:
    1. Encoding issues (utf-8, utf-8-sig, latin1, cp1252)
    2. YAML syntax issues (true: -> on:)
    3. Structure validation
    """
    
    workflow_dir = Path('.github/workflows')
    if not workflow_dir.exists():
        print("❌ .github/workflows directory not found")
        return
    
    files = list(workflow_dir.glob('*.yml'))
    fixed_count = 0
    
    for file_path in files:
        try:
            # Try multiple encodings to read the file
            content = None
            encoding_used = None
            
            for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    encoding_used = encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                print(f"❌ Could not read {file_path.name} with any encoding")
                continue
            
            # Fix the YAML syntax issues
            original_content = content
            
            # Replace 'true:' with 'on:' at the beginning of lines
            content = re.sub(r'^(\s*)true:\s*$', r'\1on:', content, flags=re.MULTILINE)
            
            # Ensure proper YAML structure
            if 'on:' not in content and 'true:' in original_content:
                # If we still have issues, force replace
                content = content.replace('true:', 'on:')
            
            # Write back with UTF-8 encoding
            with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(content)
            
            # Validate the YAML
            try:
                yaml_data = yaml.safe_load(content)
                if isinstance(yaml_data, dict) and 'on' in yaml_data and 'name' in yaml_data:
                    print(f"✅ Fixed and validated: {file_path.name}")
                    fixed_count += 1
                else:
                    print(f"⚠️  Fixed but structure issues: {file_path.name}")
            except yaml.YAMLError as e:
                print(f"⚠️  Fixed encoding but YAML error in {file_path.name}: {e}")
                
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Files processed: {len(files)}")
    print(f"   Successfully fixed: {fixed_count}")
    print(f"   Success rate: {(fixed_count/len(files)*100):.1f}%")

if __name__ == "__main__":
    print("🔧 Starting robust workflow fix...")
    fix_workflows()
    print("✅ Workflow fix complete!")
