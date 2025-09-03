#!/usr/bin/env python3
"""
Fix workflow structure issues - move misplaced 'on:' sections to the top
"""

import os
import re
from pathlib import Path

def fix_workflow_structure():
    """Fix workflows with misplaced 'on:' sections"""
    workflow_path = Path('.github/workflows')
    
    if not workflow_path.exists():
        print("❌ No .github/workflows directory found")
        return
    
    workflow_files = list(workflow_path.glob('*.yml')) + list(workflow_path.glob('*.yaml'))
    fixed_count = 0
    
    for wf_file in workflow_files:
        try:
            with open(wf_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if 'on': appears after the first job definition
            if "'on':" in content:
                lines = content.split('\n')
                
                # Find where the misplaced 'on:' section starts
                on_section_start = -1
                on_section_lines = []
                
                for i, line in enumerate(lines):
                    if line.strip() == "'on':":
                        on_section_start = i
                        on_section_lines.append('on:')  # Remove quotes
                        
                        # Collect the rest of the 'on:' section
                        j = i + 1
                        while j < len(lines):
                            next_line = lines[j]
                            if next_line.strip() and not next_line.startswith(' ') and not next_line.startswith('\t'):
                                break
                            on_section_lines.append(next_line)
                            j += 1
                        
                        # Remove the misplaced 'on:' section
                        lines = lines[:i] + lines[j:]
                        break
                
                if on_section_start != -1:
                    # Find where to insert the 'on:' section (after name, before permissions/jobs)
                    insert_pos = 1  # After name line
                    
                    for i, line in enumerate(lines):
                        if line.startswith('name:'):
                            insert_pos = i + 1
                            break
                    
                    # Insert the 'on:' section at the correct position
                    new_lines = lines[:insert_pos] + on_section_lines + lines[insert_pos:]
                    
                    # Write the fixed content
                    with open(wf_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(new_lines))
                    
                    print(f"✅ Fixed {wf_file.name}")
                    fixed_count += 1
                
        except Exception as e:
            print(f"❌ Error fixing {wf_file.name}: {e}")
    
    print(f"\n🔧 Fixed {fixed_count} workflow structure issues")

if __name__ == '__main__':
    fix_workflow_structure()
