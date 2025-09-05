#!/usr/bin/env python3
"""
Fix the remaining 10 workflows with EST timezone issues
"""

import os
import re

WORKFLOWS_DIR = r"c:\Users\kevin\Downloads\C# ai bot\.github\workflows"

def fix_remaining_workflows():
    """Fix the specific workflows identified as having EST issues"""
    
    fixes = [
        # daily_consolidated.yml - Fix UTC morning daily
        {
            'file': 'daily_consolidated.yml',
            'old': "- cron: '0 8 * * *'     # 4:00 AM EST daily",
            'new': "- cron: '0 12 * * *'    # 8:00 AM EST daily"
        },
        
        # failed_patterns.yml - Already fixed, but double-check
        {
            'file': 'failed_patterns.yml', 
            'old': "*/30 6-9:30,16-20 * * 1-5",
            'new': "*/30 10-13,20-23 * * 1-5"  # Pre/post market EST
        },
        
        # intermarket.yml - Fix early UTC hours
        {
            'file': 'intermarket.yml',
            'old': "*/20 6-22 * * *",
            'new': "*/20 10-2 * * *"  # 6AM-10PM EST (simplified)
        },
        
        # opex_calendar.yml - Fix UTC morning daily  
        {
            'file': 'opex_calendar.yml',
            'old': "0 9 * * 1,3,5",
            'new': "0 13 * * 1,3,5"  # 9AM EST = 13 UTC
        },
        
        # options_flow.yml - Fix early UTC hours
        {
            'file': 'options_flow.yml',
            'old': "*/30 8-13,21-22 * * 1-5",
            'new': "*/30 12-17,1-2 * * 1-5"  # 8AM-1PM EST, 9PM-10PM EST
        },
        
        # portfolio_heat.yml - Fix early UTC hours  
        {
            'file': 'portfolio_heat.yml',
            'old': "*/20 8-13,21-23 * * 1-5",
            'new': "*/20 12-17,1-3 * * 1-5"  # 8AM-1PM EST, 9PM-11PM EST
        },
        
        # seasonality.yml - Fix UTC morning daily
        {
            'file': 'seasonality.yml', 
            'old': "0 6 * * 1",
            'new': "0 10 * * 1"  # 6AM EST = 10 UTC
        },
        
        # ultimate_ml_rl_intel_system.yml - Fix early UTC hours
        {
            'file': 'ultimate_ml_rl_intel_system.yml',
            'old': "*/10 8-13 * * 1-5",
            'new': "*/10 12-17 * * 1-5"  # 8AM-1PM EST
        },
        {
            'file': 'ultimate_ml_rl_intel_system.yml',
            'old': "*/30 0-8,23 * * *", 
            'new': "*/30 4-12 * * *"  # Midnight-8AM EST = 4-12 UTC
        },
        
        # zones_identifier.yml - Fix UTC morning daily
        {
            'file': 'zones_identifier.yml',
            'old': "0 6 * * 0,6",
            'new': "0 10 * * 0,6"  # 6AM EST = 10 UTC
        }
    ]
    
    fixed_files = set()
    
    for fix in fixes:
        filepath = os.path.join(WORKFLOWS_DIR, fix['file'])
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if fix['old'] in content:
                content = content.replace(fix['old'], fix['new'])
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"✅ Fixed: {fix['file']}")
                fixed_files.add(fix['file'])
            else:
                print(f"⏭️  Skipped: {fix['file']} (pattern not found)")
                
        except Exception as e:
            print(f"❌ Error fixing {fix['file']}: {e}")
    
    print(f"\n🎯 Fixed {len(fixed_files)} workflow files for EST timezone!")
    return fixed_files

if __name__ == "__main__":
    print("🔧 Fixing remaining workflows for EST timezone compliance...")
    print("=" * 60)
    fix_remaining_workflows()
