#!/usr/bin/env python3
"""
Quick Workflow Schedule Validation
Tests if workflows have proper schedule syntax and will trigger
"""

import yaml
import os
from datetime import datetime, timezone
from croniter import croniter

def validate_workflow_schedules():
    """Validate all workflow schedules"""
    print("🔍 VALIDATING WORKFLOW SCHEDULES")
    print("=" * 50)
    
    workflow_dir = ".github/workflows"
    valid_schedules = 0
    invalid_schedules = 0
    no_schedules = 0
    
    for filename in os.listdir(workflow_dir):
        if filename.endswith('.yml') or filename.endswith('.yaml'):
            filepath = os.path.join(workflow_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = yaml.safe_load(f)
                
                workflow_name = content.get('name', filename)
                on_config = content.get('on', {})
                
                if 'schedule' in on_config:
                    schedules = on_config['schedule']
                    print(f"✅ {workflow_name}")
                    
                    for i, schedule in enumerate(schedules):
                        cron = schedule.get('cron', '')
                        if cron:
                            try:
                                # Test if cron is valid
                                cron_iter = croniter(cron, datetime.now(timezone.utc))
                                next_run = cron_iter.get_next(datetime)
                                print(f"   Schedule {i+1}: {cron}")
                                print(f"   Next run: {next_run.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                                valid_schedules += 1
                            except Exception as e:
                                print(f"   ❌ Invalid cron: {cron} - {e}")
                                invalid_schedules += 1
                    print()
                else:
                    print(f"⚠️  {workflow_name} - No schedule found")
                    no_schedules += 1
                    
            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")
                invalid_schedules += 1
    
    print("📊 VALIDATION SUMMARY:")
    print(f"✅ Valid schedules: {valid_schedules}")
    print(f"❌ Invalid schedules: {invalid_schedules}")  
    print(f"⚠️  No schedules: {no_schedules}")
    
    if invalid_schedules == 0:
        print("🎉 ALL SCHEDULES ARE VALID!")
        return True
    else:
        print("💥 SOME SCHEDULES HAVE ISSUES!")
        return False

if __name__ == "__main__":
    validate_workflow_schedules()
