#!/usr/bin/env python3
"""
Precise YAML Workflow Fixer
Fixes indentation and structure issues in workflow files
"""

import os
import re
import yaml

def fix_yaml_workflow(filepath):
    """Fix YAML indentation and structure issues"""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix the specific issue where 'with:' is not indented under 'uses:'
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # If we find a 'uses:' line
        if re.match(r'\s*-\s*uses:', line):
            fixed_lines.append(line)
            
            # Check if the next line is 'with:' without proper indentation
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if re.match(r'\s*with:', next_line) and not re.match(r'\s*-\s*with:', next_line):
                    # Get the indentation of the 'uses:' line
                    uses_indent = len(line) - len(line.lstrip())
                    # Add proper indentation for 'with:' (should be indented more than 'uses:')
                    with_indent = ' ' * (uses_indent + 2)
                    fixed_with_line = with_indent + next_line.strip()
                    fixed_lines.append(fixed_with_line)
                    i += 2
                    
                    # Fix subsequent lines in the 'with:' block
                    while i < len(lines):
                        current = lines[i]
                        if current.strip() == '':
                            fixed_lines.append(current)
                            i += 1
                        elif re.match(r'\s*[a-zA-Z_-]+:', current) and not re.match(r'\s*-', current):
                            # This is a property of the 'with:' block
                            prop_indent = ' ' * (uses_indent + 4)
                            fixed_prop_line = prop_indent + current.strip()
                            fixed_lines.append(fixed_prop_line)
                            i += 1
                        else:
                            # We've reached the next step or section
                            break
                else:
                    i += 1
            else:
                i += 1
        else:
            fixed_lines.append(line)
            i += 1
    
    # Join the lines back together
    fixed_content = '\n'.join(fixed_lines)
    
    # Test if the YAML is now valid
    try:
        yaml.safe_load(fixed_content)
        return fixed_content
    except yaml.YAMLError as e:
        print(f"YAML still invalid after fix: {e}")
        return content  # Return original if fix didn't work

def fix_all_workflows():
    """Fix all workflow files"""
    
    workflow_dir = ".github/workflows"
    fixed_count = 0
    
    print("🔧 Fixing YAML indentation issues in all workflows...")
    
    for filename in os.listdir(workflow_dir):
        if filename.endswith('.yml') or filename.endswith('.yaml'):
            filepath = os.path.join(workflow_dir, filename)
            
            try:
                # Try to load the original YAML
                with open(filepath, 'r') as f:
                    content = f.read()
                
                try:
                    yaml.safe_load(content)
                    print(f"  ✅ {filename}: Already valid")
                    continue
                except yaml.YAMLError:
                    print(f"  🔧 {filename}: Fixing YAML issues...")
                    
                    # Create backup
                    with open(f"{filepath}.backup", 'w') as f:
                        f.write(content)
                    
                    # Fix the YAML
                    fixed_content = fix_yaml_workflow(filepath)
                    
                    # Write the fixed content
                    with open(filepath, 'w') as f:
                        f.write(fixed_content)
                    
                    # Verify the fix worked
                    try:
                        yaml.safe_load(fixed_content)
                        print(f"  ✅ {filename}: Fixed successfully")
                        fixed_count += 1
                    except yaml.YAMLError as e:
                        print(f"  ❌ {filename}: Still has issues: {e}")
                        # Restore backup if fix didn't work
                        with open(f"{filepath}.backup", 'r') as f:
                            original = f.read()
                        with open(filepath, 'w') as f:
                            f.write(original)
                        
            except Exception as e:
                print(f"  ❌ {filename}: Error processing file: {e}")
    
    print(f"\n✅ Fixed {fixed_count} workflow files")
    return fixed_count

if __name__ == "__main__":
    fixed_count = fix_all_workflows()
    
    print(f"\n🧪 Running validation on all workflows...")
    
    # Run validation
    workflow_dir = ".github/workflows"
    valid_count = 0
    total_count = 0
    
    for filename in os.listdir(workflow_dir):
        if filename.endswith('.yml') or filename.endswith('.yaml'):
            total_count += 1
            filepath = os.path.join(workflow_dir, filename)
            
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                yaml.safe_load(content)
                valid_count += 1
                print(f"  ✅ {filename}")
            except yaml.YAMLError as e:
                print(f"  ❌ {filename}: {str(e)[:100]}...")
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"  Total Workflows: {total_count}")
    print(f"  Valid YAML: {valid_count}/{total_count}")
    print(f"  Success Rate: {(valid_count/total_count)*100:.1f}%")
    
    if valid_count == total_count:
        print(f"  🎉 ALL WORKFLOWS ARE NOW VALID!")
    else:
        print(f"  ⚠️  {total_count - valid_count} workflows still need attention")