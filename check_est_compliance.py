#!/usr/bin/env python3
"""
Check all 27 GitHub Actions workflows for EST timezone compliance
"""

import os
import re
import glob

# Define the workflow directory
WORKFLOWS_DIR = r"c:\Users\kevin\Downloads\C# ai bot\.github\workflows"

def analyze_cron_schedule(content, filename):
    """Analyze cron schedules to detect non-EST patterns"""
    
    # Find all cron patterns
    cron_patterns = re.findall(r"cron: '([^']+)'", content)
    
    issues = []
    est_compliant = []
    
    for cron in cron_patterns:
        # Check for common UTC patterns that should be EST
        if re.search(r'\*/\d+ [0-8]-', cron):  # Early morning UTC hours
            issues.append(f"❌ Early UTC hours: {cron}")
        elif re.search(r'\*/\d+ [4-9]-1[0-6]', cron):  # Business hours in UTC
            issues.append(f"❌ UTC business hours: {cron}")
        elif re.search(r'0 [4-9] \* \*', cron):  # Daily times in UTC morning
            issues.append(f"❌ UTC morning daily: {cron}")
        else:
            est_compliant.append(f"✅ Looks EST compliant: {cron}")
    
    return issues, est_compliant

def check_workflow_file(filepath):
    """Check a single workflow file for EST compliance"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        filename = os.path.basename(filepath)
        issues, compliant = analyze_cron_schedule(content, filename)
        
        return {
            'filename': filename,
            'issues': issues,
            'compliant': compliant,
            'total_schedules': len(issues) + len(compliant)
        }
            
    except Exception as e:
        return {
            'filename': os.path.basename(filepath),
            'error': str(e),
            'issues': [],
            'compliant': [],
            'total_schedules': 0
        }

def main():
    """Check all workflow files for EST compliance"""
    print("🔍 Checking EST timezone compliance for all 27 workflows...")
    print("=" * 70)
    
    # Get all workflow files
    workflow_files = glob.glob(os.path.join(WORKFLOWS_DIR, "*.yml"))
    
    if not workflow_files:
        print(f"❌ No workflow files found in {WORKFLOWS_DIR}")
        return
    
    print(f"📁 Found {len(workflow_files)} workflow files")
    print()
    
    total_issues = 0
    files_with_issues = 0
    files_compliant = 0
    
    for filepath in sorted(workflow_files):
        result = check_workflow_file(filepath)
        
        if 'error' in result:
            print(f"❌ ERROR: {result['filename']} - {result['error']}")
            continue
        
        print(f"📄 {result['filename']}")
        
        if result['issues']:
            files_with_issues += 1
            total_issues += len(result['issues'])
            for issue in result['issues']:
                print(f"    {issue}")
        
        if result['compliant']:
            for comp in result['compliant']:
                print(f"    {comp}")
        
        if not result['issues'] and not result['compliant']:
            print(f"    ℹ️  No scheduled runs (manual trigger only)")
        
        if not result['issues'] and result['compliant']:
            files_compliant += 1
        
        print()
    
    print("=" * 70)
    print(f"📊 SUMMARY:")
    print(f"✅ Files EST compliant: {files_compliant}")
    print(f"⚠️  Files needing fixes: {files_with_issues}")
    print(f"🔧 Total issues found: {total_issues}")
    
    if total_issues > 0:
        print(f"\n🎯 ACTION NEEDED: {files_with_issues} workflow files need EST timezone fixes!")
    else:
        print(f"\n🎉 ALL GOOD: All workflows are EST compliant!")

if __name__ == "__main__":
    main()
