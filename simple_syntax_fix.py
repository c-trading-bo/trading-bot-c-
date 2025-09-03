#!/usr/bin/env python3
"""
SIMPLE ROBUST FIX - Fix the actual syntax errors in workflow files
"""

import os
from pathlib import Path

def fix_syntax_errors():
    """Fix the actual syntax errors in workflow files"""
    workflow_path = Path('.github/workflows')
    
    if not workflow_path.exists():
        print("❌ No .github/workflows directory found")
        return
    
    workflow_files = list(workflow_path.glob('*.yml')) + list(workflow_path.glob('*.yaml'))
    
    print(f"🔧 FIXING SYNTAX ERRORS IN {len(workflow_files)} WORKFLOWS...\n")
    
    fixed_count = 0
    
    for wf_file in workflow_files:
        print(f"📄 {wf_file.name}:", end=" ")
        
        try:
            # Read the file
            with open(wf_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Fix the main syntax error: "true:" -> "on:"
            if '\ntrue:\n' in content:
                content = content.replace('\ntrue:\n', '\non:\n')
                print("Fixed true: -> on:", end=" ")
            elif content.startswith('true:\n'):
                content = content.replace('true:\n', 'on:\n', 1)
                print("Fixed true: -> on:", end=" ")
            
            # Also fix if there are spaces
            if '\ntrue: \n' in content:
                content = content.replace('\ntrue: \n', '\non:\n')
                print("Fixed true: -> on:", end=" ")
            
            # Write back if changed
            if content != original_content:
                with open(wf_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print("✅ FIXED")
                fixed_count += 1
            else:
                print("✅ OK")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print(f"\n📊 SYNTAX FIX SUMMARY:")
    print(f"   🔧 Fixed: {fixed_count}/{len(workflow_files)}")
    
    # Validate the fixes
    print(f"\n🔍 VALIDATING FIXES...")
    
    error_count = 0
    for wf_file in workflow_files:
        try:
            with open(wf_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for syntax errors
            if '\ntrue:\n' in content or content.startswith('true:\n'):
                print(f"   ❌ {wf_file.name} still has 'true:' syntax error")
                error_count += 1
            elif 'runs-on: ubuntu-latest' not in content:
                print(f"   ⚠️ {wf_file.name} might be missing required content")
                
        except Exception as e:
            print(f"   ❌ {wf_file.name}: {e}")
            error_count += 1
    
    if error_count == 0:
        print("   ✅ All files validated successfully!")
    else:
        print(f"   ⚠️ {error_count} files still have issues")

if __name__ == '__main__':
    fix_syntax_errors()
