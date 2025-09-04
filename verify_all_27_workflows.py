#!/usr/bin/env python3
"""
COMPREHENSIVE VERIFICATION: ALL 27 WORKFLOWS
Verify every single workflow has proper scheduling and show countdown to next run
"""

import os
import yaml
import re
from datetime import datetime, timedelta
import pytz
from croniter import croniter

def parse_cron_schedule(cron_expr):
    """Parse cron expression and return next run time"""
    try:
        # Current time in UTC (GitHub Actions runs in UTC)
        now = datetime.now(pytz.UTC)
        
        # Create croniter object
        cron = croniter(cron_expr, now)
        next_run = cron.get_next(datetime)
        
        # Calculate time until next run
        time_until = next_run - now
        
        return {
            'next_run': next_run.strftime('%Y-%m-%d %H:%M:%S UTC'),
            'time_until': str(time_until).split('.')[0],  # Remove microseconds
            'valid': True
        }
    except Exception as e:
        return {
            'next_run': f'ERROR: {str(e)}',
            'time_until': 'INVALID CRON',
            'valid': False
        }

def extract_schedule_from_workflow(file_path):
    """Extract schedule information from workflow file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Parse YAML
        try:
            workflow = yaml.safe_load(content)
        except yaml.YAMLError:
            return {'error': 'Invalid YAML format'}
            
        # Check for schedule in 'on' section
        if 'on' not in workflow:
            return {'error': 'No "on" section found'}
            
        on_section = workflow['on']
        
        # Handle different formats
        if isinstance(on_section, dict):
            if 'schedule' in on_section:
                schedules = on_section['schedule']
            else:
                return {'error': 'No schedule section found'}
        elif isinstance(on_section, list):
            # Find schedule in list
            schedule_found = False
            for item in on_section:
                if isinstance(item, dict) and 'schedule' in item:
                    schedules = item['schedule']
                    schedule_found = True
                    break
            if not schedule_found:
                return {'error': 'No schedule found in on list'}
        else:
            return {'error': 'Unexpected on section format'}
            
        if not schedules:
            return {'error': 'Empty schedule section'}
            
        # Parse each cron schedule
        cron_schedules = []
        for schedule_item in schedules:
            if isinstance(schedule_item, dict) and 'cron' in schedule_item:
                cron_expr = schedule_item['cron']
                cron_info = parse_cron_schedule(cron_expr)
                cron_info['expression'] = cron_expr
                cron_schedules.append(cron_info)
                
        if not cron_schedules:
            return {'error': 'No valid cron schedules found'}
            
        # Find the next upcoming schedule
        valid_schedules = [s for s in cron_schedules if s['valid']]
        if not valid_schedules:
            return {'error': 'No valid cron schedules'}
            
        # Sort by next run time and get the soonest
        valid_schedules.sort(key=lambda x: x['next_run'])
        next_schedule = valid_schedules[0]
        
        return {
            'total_schedules': len(cron_schedules),
            'valid_schedules': len(valid_schedules),
            'next_run': next_schedule['next_run'],
            'time_until': next_schedule['time_until'],
            'next_cron': next_schedule['expression'],
            'all_schedules': cron_schedules
        }
        
    except Exception as e:
        return {'error': f'Exception reading file: {str(e)}'}

def check_botcore_integration(file_path):
    """Check if workflow has BotCore integration"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Look for BotCore integration indicators
        integration_indicators = [
            'BotCore Decision Engine',
            'workflow_data_integration.py',
            'Intelligence/data/integrated/',
            'BotCore.*integration'
        ]
        
        has_integration = False
        for indicator in integration_indicators:
            if re.search(indicator, content, re.IGNORECASE):
                has_integration = True
                break
                
        return has_integration
    except:
        return False

def main():
    """Main verification function"""
    print("🔍 COMPREHENSIVE VERIFICATION: ALL 27 WORKFLOWS")
    print("=" * 80)
    print(f"Verification Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    workflows_dir = ".github/workflows"
    
    # Get all workflow files
    workflow_files = []
    for file in os.listdir(workflows_dir):
        if file.endswith('.yml') or file.endswith('.yaml'):
            workflow_files.append(file)
    
    workflow_files.sort()
    
    print(f"📊 FOUND {len(workflow_files)} WORKFLOW FILES")
    print()
    
    # Verify each workflow
    results = []
    
    for i, workflow_file in enumerate(workflow_files, 1):
        print(f"🔍 [{i:2d}/27] Checking: {workflow_file}")
        
        file_path = os.path.join(workflows_dir, workflow_file)
        
        # Check schedule
        schedule_info = extract_schedule_from_workflow(file_path)
        
        # Check BotCore integration
        has_integration = check_botcore_integration(file_path)
        
        result = {
            'file': workflow_file,
            'has_schedule': 'error' not in schedule_info,
            'has_integration': has_integration,
            'schedule_info': schedule_info
        }
        results.append(result)
        
        # Print immediate status
        schedule_status = "✅" if result['has_schedule'] else "❌"
        integration_status = "✅" if result['has_integration'] else "❌"
        
        print(f"    Schedule: {schedule_status} | BotCore: {integration_status}")
        
        if result['has_schedule']:
            print(f"    Next Run: {schedule_info['next_run']}")
            print(f"    Time Until: {schedule_info['time_until']}")
        else:
            print(f"    Error: {schedule_info.get('error', 'Unknown error')}")
        print()
    
    # Summary statistics
    print("=" * 80)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 80)
    
    total_workflows = len(results)
    scheduled_workflows = sum(1 for r in results if r['has_schedule'])
    integrated_workflows = sum(1 for r in results if r['has_integration'])
    fully_complete = sum(1 for r in results if r['has_schedule'] and r['has_integration'])
    
    print(f"Total Workflows Found: {total_workflows}")
    print(f"With Valid Scheduling: {scheduled_workflows}/{total_workflows}")
    print(f"With BotCore Integration: {integrated_workflows}/{total_workflows}")
    print(f"Fully Complete (Schedule + Integration): {fully_complete}/{total_workflows}")
    print()
    
    # Show next run times for all scheduled workflows
    print("⏰ NEXT RUN TIMES FOR ALL SCHEDULED WORKFLOWS")
    print("=" * 80)
    
    scheduled_results = [r for r in results if r['has_schedule']]
    scheduled_results.sort(key=lambda x: x['schedule_info']['next_run'])
    
    for i, result in enumerate(scheduled_results, 1):
        file_name = result['file'].replace('.yml', '')
        integration_indicator = "✅" if result['has_integration'] else "❌"
        
        print(f"{i:2d}. {file_name:<35} | Next: {result['schedule_info']['next_run']} | Until: {result['schedule_info']['time_until']} | BotCore: {integration_indicator}")
    
    # Show problematic workflows
    problematic = [r for r in results if not r['has_schedule'] or not r['has_integration']]
    
    if problematic:
        print()
        print("⚠️  PROBLEMATIC WORKFLOWS")
        print("=" * 80)
        
        for result in problematic:
            issues = []
            if not result['has_schedule']:
                issues.append("NO SCHEDULE")
            if not result['has_integration']:
                issues.append("NO BOTCORE INTEGRATION")
                
            print(f"❌ {result['file']:<35} | Issues: {', '.join(issues)}")
            
            if not result['has_schedule'] and 'error' in result['schedule_info']:
                print(f"   Schedule Error: {result['schedule_info']['error']}")
    
    print()
    print("=" * 80)
    
    if total_workflows == 27 and fully_complete == 27:
        print("🎉 SUCCESS: ALL 27 WORKFLOWS ARE FULLY CONFIGURED!")
        print("✅ 100% Scheduling Coverage")
        print("✅ 100% BotCore Integration")
        print("🚀 Your trading bot has maximum intelligence!")
    else:
        print(f"⚠️  ATTENTION REQUIRED:")
        if total_workflows != 27:
            print(f"   Expected 27 workflows, found {total_workflows}")
        if scheduled_workflows != 27:
            print(f"   {27 - scheduled_workflows} workflows missing scheduling")
        if integrated_workflows != 27:
            print(f"   {27 - integrated_workflows} workflows missing BotCore integration")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
