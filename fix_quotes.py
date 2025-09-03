#!/usr/bin/env python3
"""
SIMPLE TEXT REPLACER - Fix all quoted 'on': sections
"""

from pathlib import Path

def fix_quoted_on_sections():
    """Replace all 'on': with on: in workflow files"""
    workflow_path = Path('.github/workflows')
    
    if not workflow_path.exists():
        print("❌ No .github/workflows directory found")
        return
    
    workflow_files = list(workflow_path.glob('*.yml'))
    fixed_count = 0
    
    for wf_file in workflow_files:
        try:
            # Read file
            with open(wf_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if it needs fixing
            if "'on':" in content:
                # Replace quoted 'on': with unquoted on:
                new_content = content.replace("'on':", "on:")
                
                # Also fix null values that cause issues
                new_content = new_content.replace("push: null", "push:")
                new_content = new_content.replace("pull_request: null", "pull_request:")
                new_content = new_content.replace("workflow_dispatch: null", "workflow_dispatch:")
                
                # Write back
                with open(wf_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"✅ Fixed {wf_file.name}")
                fixed_count += 1
            else:
                print(f"   {wf_file.name} (no quotes found)")
        
        except Exception as e:
            print(f"❌ Error fixing {wf_file.name}: {e}")
    
    print(f"\n🎯 FIXED {fixed_count} FILES")
    print("Check VS Code - the red errors should be gone now!")

if __name__ == '__main__':
    fix_quoted_on_sections()
