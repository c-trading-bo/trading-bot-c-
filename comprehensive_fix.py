#!/usr/bin/env python3
"""
COMPREHENSIVE GITHUB ACTIONS FIXER
Fix ALL common GitHub Actions validation issues
"""

import yaml
from pathlib import Path

def comprehensive_github_actions_fix():
    """Fix all GitHub Actions validation issues"""
    workflow_path = Path('.github/workflows')
    
    if not workflow_path.exists():
        print("❌ No .github/workflows directory found")
        return
    
    workflow_files = list(workflow_path.glob('*.yml'))
    
    print(f"🔧 COMPREHENSIVE FIX FOR {len(workflow_files)} WORKFLOWS...\n")
    
    for wf_file in workflow_files:
        print(f"📄 {wf_file.name}:")
        fixed_issues = []
        
        try:
            # Read and parse
            with open(wf_file, 'r', encoding='utf-8') as f:
                workflow = yaml.safe_load(f)
            
            if not workflow:
                print("  ❌ Empty workflow")
                continue
            
            # Fix missing required fields
            if 'name' not in workflow:
                workflow['name'] = wf_file.stem.replace('_', ' ').title()
                fixed_issues.append("Added missing name")
            
            if 'on' not in workflow:
                workflow['on'] = {'workflow_dispatch': {}}
                fixed_issues.append("Added missing 'on' section")
            
            if 'jobs' not in workflow:
                workflow['jobs'] = {}
                fixed_issues.append("Added missing jobs section")
            
            # Fix jobs that are missing required fields
            if workflow['jobs']:
                for job_name, job_config in workflow['jobs'].items():
                    if not isinstance(job_config, dict):
                        workflow['jobs'][job_name] = {'runs-on': 'ubuntu-latest', 'steps': []}
                        fixed_issues.append(f"Fixed malformed job {job_name}")
                        continue
                    
                    # Add missing runs-on
                    if 'runs-on' not in job_config:
                        workflow['jobs'][job_name]['runs-on'] = 'ubuntu-latest'
                        fixed_issues.append(f"Added runs-on to {job_name}")
                    
                    # Add missing steps (if completely missing)
                    if 'steps' not in job_config:
                        workflow['jobs'][job_name]['steps'] = [
                            {'uses': 'actions/checkout@v4'},
                            {'run': 'echo "Job needs configuration"'}
                        ]
                        fixed_issues.append(f"Added basic steps to {job_name}")
                    elif not job_config['steps']:
                        workflow['jobs'][job_name]['steps'] = [
                            {'uses': 'actions/checkout@v4'},
                            {'run': 'echo "Job needs configuration"'}
                        ]
                        fixed_issues.append(f"Added steps to empty {job_name}")
            
            # Write the fixed workflow
            if fixed_issues:
                with open(wf_file, 'w', encoding='utf-8') as f:
                    yaml.dump(workflow, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                
                for issue in fixed_issues:
                    print(f"  ✅ {issue}")
            else:
                print("  ✅ No issues found")
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n🎯 COMPREHENSIVE FIX COMPLETE!")
    print("All workflows should now pass GitHub Actions validation!")

if __name__ == '__main__':
    comprehensive_github_actions_fix()
