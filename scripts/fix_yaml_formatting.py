#!/usr/bin/env python3
"""
🔧 CRITICAL FIX: Repair YAML formatting issues from bulletproof script
"""

import os
from pathlib import Path
import re

def fix_yaml_formatting():
    """Fix YAML formatting issues in all workflows"""
    
    workflow_dir = Path('.github/workflows')
    
    print("🔧 FIXING YAML FORMATTING ISSUES...")
    
    for workflow_file in workflow_dir.glob('*.yml'):
        if workflow_file.name == 'test_optimization.yml':
            continue
            
        print(f"🔧 Fixing: {workflow_file.name}")
        
        with open(workflow_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove the problematic bulletproof error handling that broke YAML
        # Find and remove the malformed error handling sections
        lines = content.split('\n')
        fixed_lines = []
        skip_until_next_step = False
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Skip the malformed bulletproof sections
            if '# 🛡️ BULLETPROOF: Python error handling' in line:
                skip_until_next_step = True
                i += 1
                continue
            
            if skip_until_next_step:
                # Skip malformed lines until we reach the actual Python code or next step
                if line.strip().startswith('- name:') or line.strip().startswith('run: |'):
                    skip_until_next_step = False
                    if line.strip().startswith('run: |'):
                        fixed_lines.append(line)
                        i += 1
                        continue
                elif line.strip().startswith('set -e') or line.strip().startswith('echo "🐍') or line.strip().startswith('python --version'):
                    i += 1
                    continue
                else:
                    skip_until_next_step = False
            
            fixed_lines.append(line)
            i += 1
        
        # Write back the fixed content
        with open(workflow_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(fixed_lines))
        
        print(f"  ✅ Fixed YAML formatting")

def add_simple_bulletproof_features():
    """Add simple, non-breaking bulletproof features"""
    
    workflow_dir = Path('.github/workflows')
    
    for workflow_file in workflow_dir.glob('*.yml'):
        if workflow_file.name == 'test_optimization.yml':
            continue
            
        with open(workflow_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Only add timeout if not present and make secrets optional
        modified = False
        
        # Fix SONAR_TOKEN to be optional
        if '${{ secrets.SONAR_TOKEN }}' in content and 'DISABLED' not in content:
            content = content.replace(
                '${{ secrets.SONAR_TOKEN }}',
                '${{ secrets.SONAR_TOKEN || \'DISABLED\' }}'
            )
            modified = True
        
        # Add timeout to jobs without it (simple approach)
        if 'runs-on:' in content and 'timeout-minutes:' not in content:
            content = content.replace(
                'runs-on: ubuntu-latest',
                'runs-on: ubuntu-latest\n    timeout-minutes: 30  # 🛡️ Prevent hanging'
            )
            modified = True
        
        if modified:
            with open(workflow_file, 'w', encoding='utf-8') as f:
                f.write(content)

if __name__ == "__main__":
    fix_yaml_formatting()
    add_simple_bulletproof_features()
    print("✅ YAML FORMATTING FIXED!")
    print("🛡️ Simple bulletproof features added!")
