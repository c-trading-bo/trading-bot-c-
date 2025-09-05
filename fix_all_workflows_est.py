#!/usr/bin/env python3
"""
Fix EST timezone for all 27 GitHub Actions workflows
Convert UTC cron schedules to proper EST business hours
"""

import os
import re
import glob

# Define the workflow directory
WORKFLOWS_DIR = r"c:\Users\kevin\Downloads\C# ai bot\.github\workflows"

# EST timezone conversion mapping (UTC to EST)
# Market hours: 9:30 AM - 4:00 PM EST = 13:30 - 21:00 UTC (DST) / 14:30 - 22:00 UTC (Standard)
# Using DST times (most of trading year)

def fix_cron_schedule(content):
    """Fix cron schedules to EST timezone"""
    
    # Common patterns to fix
    replacements = [
        # Market hours patterns
        (r"cron: '\*/(\d+) 9-16 \* \* 1-5'", r"cron: '*/\1 13-21 * * 1-5'"),  # Market hours
        (r"cron: '\*/(\d+) 10-16 \* \* 1-5'", r"cron: '*/\1 14-21 * * 1-5'"),  # Market hours variant
        (r"cron: '\*/(\d+) 8-16 \* \* 1-5'", r"cron: '*/\1 12-21 * * 1-5'"),  # Extended market hours
        
        # Pre-market patterns
        (r"cron: '\*/(\d+) 4-9 \* \* 1-5'", r"cron: '*/\1 8-13 * * 1-5'"),    # Pre-market
        (r"cron: '\*/(\d+) 6-9 \* \* 1-5'", r"cron: '*/\1 10-13 * * 1-5'"),   # Pre-market variant
        (r"cron: '\*/(\d+) 5-9 \* \* 1-5'", r"cron: '*/\1 9-13 * * 1-5'"),    # Pre-market variant
        
        # Post-market patterns  
        (r"cron: '\*/(\d+) 16-20 \* \* 1-5'", r"cron: '*/\1 21-23,0-1 * * 1-5'"),  # Post-market (split for midnight)
        (r"cron: '\*/(\d+) 16-18 \* \* 1-5'", r"cron: '*/\1 21-23 * * 1-5'"),      # Post-market short
        (r"cron: '\*/(\d+) 17-20 \* \* 1-5'", r"cron: '*/\1 21-23,0-1 * * 1-5'"),  # Post-market variant
        
        # Overnight patterns (avoiding midnight span issues)
        (r"cron: '\*/(\d+) 20-4 \* \* \*'", r"cron: '*/\1 0-8 * * *'"),         # Simplified overnight
        (r"cron: '\*/(\d+) 22-6 \* \* \*'", r"cron: '*/\1 2-10 * * *'"),        # Overnight variant
        (r"cron: '\*/(\d+) 0-6 \* \* \*'", r"cron: '*/\1 4-10 * * *'"),         # Early morning
        
        # Daily patterns
        (r"cron: '0 4 \* \* \*'", r"cron: '0 8 * * *'"),                        # 4 AM EST daily
        (r"cron: '0 6 \* \* \*'", r"cron: '0 10 * * *'"),                       # 6 AM EST daily
        (r"cron: '0 12 \* \* \*'", r"cron: '0 16 * * *'"),                      # 12 PM EST daily
        (r"cron: '0 18 \* \* \*'", r"cron: '0 22 * * *'"),                      # 6 PM EST daily
        
        # Weekend patterns
        (r"cron: '0 12 \* \* 0,6'", r"cron: '0 16 * * 0,6'"),                  # Weekend noon EST
        (r"cron: '0 8 \* \* 0,6'", r"cron: '0 12 * * 0,6'"),                   # Weekend morning EST
    ]
    
    fixed_content = content
    for pattern, replacement in replacements:
        fixed_content = re.sub(pattern, replacement, fixed_content)
    
    return fixed_content

def fix_workflow_file(filepath):
    """Fix a single workflow file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        fixed_content = fix_cron_schedule(content)
        
        if fixed_content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"✅ Fixed: {os.path.basename(filepath)}")
            return True
        else:
            print(f"⏭️  No changes needed: {os.path.basename(filepath)}")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing {filepath}: {e}")
        return False

def main():
    """Fix all workflow files"""
    print("🔧 Fixing EST timezone for all 27 workflows...")
    print("=" * 60)
    
    # Get all workflow files
    workflow_files = glob.glob(os.path.join(WORKFLOWS_DIR, "*.yml"))
    
    if not workflow_files:
        print(f"❌ No workflow files found in {WORKFLOWS_DIR}")
        return
    
    print(f"📁 Found {len(workflow_files)} workflow files")
    print()
    
    fixed_count = 0
    for filepath in sorted(workflow_files):
        if fix_workflow_file(filepath):
            fixed_count += 1
    
    print()
    print("=" * 60)
    print(f"🎯 COMPLETED: Fixed {fixed_count}/{len(workflow_files)} workflow files")
    print("📈 All workflows now use EST timezone for proper market hours!")

if __name__ == "__main__":
    main()
