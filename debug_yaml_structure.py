#!/usr/bin/env python3
import os
import re
import yaml
from pathlib import Path

def debug_yaml_keys():
    """Debug what's causing the True key in YAML files"""
    
    workflow_dir = Path('.github/workflows')
    sample_file = workflow_dir / 'cloud_bot_mechanic.yml'
    
    print("🔍 DEBUGGING YAML STRUCTURE...")
    
    with open(sample_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Raw content preview:")
    lines = content.split('\n')
    for i, line in enumerate(lines[:50], 1):
        if ':' in line and not line.strip().startswith('#'):
            print(f"{i:2d}: {repr(line)}")
    
    print("\nParsing YAML...")
    try:
        data = yaml.safe_load(content)
        print(f"Top level keys: {list(data.keys())}")
        
        for key in data.keys():
            print(f"Key type: {type(key).__name__} = {repr(key)}")
            if key is True:
                print(f"  TRUE KEY FOUND! Value: {data[key]}")
                
    except Exception as e:
        print(f"YAML Error: {e}")

def fix_yaml_structure():
    """Fix YAML structure issues more aggressively"""
    
    workflow_dir = Path('.github/workflows')
    files = list(workflow_dir.glob('*.yml'))
    
    print(f"\n🔧 AGGRESSIVE YAML STRUCTURE FIX...")
    
    for file_path in files:
        try:
            # Read with encoding handling
            content = None
            for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                print(f"❌ Could not read {file_path.name}")
                continue
            
            # Apply multiple fixes
            original = content
            
            # Fix 1: true: -> on:
            content = re.sub(r'^(\s*)true:\s*$', r'\\1on:', content, flags=re.MULTILINE)
            
            # Fix 2: Ensure proper structure
            if not re.search(r'^on:\s*$', content, re.MULTILINE):
                # If no 'on:' found, try to fix it
                content = re.sub(r'^(\s*)true(\s*:?\s*)$', r'\\1on:\\2', content, flags=re.MULTILINE)
            
            # Fix 3: Handle any remaining True references that aren't strings
            lines = content.split('\n')
            fixed_lines = []
            
            for line in lines:
                # Skip lines that are comments or contain 'true' in strings
                if (line.strip().startswith('#') or 
                    "== 'true'" in line or 
                    '= true' in line or
                    '\"true\"' in line or
                    "'true'" in line):
                    fixed_lines.append(line)
                    continue
                
                # Fix standalone True at start of line
                if re.match(r'^\\s*True\\s*:', line):
                    line = re.sub(r'^(\\s*)True(\\s*:)', r'\\1on\\2', line)
                
                fixed_lines.append(line)
            
            content = '\n'.join(fixed_lines)
            
            # Write back as UTF-8
            with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(content)
            
            # Test parse
            try:
                data = yaml.safe_load(content)
                has_true_key = True in data if isinstance(data, dict) else False
                if not has_true_key and 'on' in data and 'name' in data:
                    print(f"✅ {file_path.name}")
                else:
                    print(f"⚠️  {file_path.name} - True key: {has_true_key}")
            except Exception as e:
                print(f"❌ {file_path.name} - YAML error: {e}")
                
        except Exception as e:
            print(f"💥 {file_path.name} - {e}")

if __name__ == "__main__":
    debug_yaml_keys()
    fix_yaml_structure()
