#!/usr/bin/env python3
"""
FIXED COMPREHENSIVE VERIFICATION: ALL 27 WORKFLOWS
Properly verify every single workflow has correct scheduling and BotCore integration
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
        now = datetime.now(pytz.UTC)
        cron = croniter(cron_expr, now)
        next_run = cron.get_next(datetime)
        time_until = next_run - now
        
        # Convert to human readable
        if time_until.total_seconds() < 3600:
            time_str = f"{int(time_until.total_seconds() // 60)} minutes"
        elif time_until.total_seconds() < 86400:
            time_str = f"{time_until.total_seconds() // 3600:.1f} hours"
        else:
            time_str = f"{time_until.days} days"
        
        return {
            'valid': True,
            'next_run': next_run.strftime('%Y-%m-%d %H:%M:%S UTC'),
            'time_until': time_str,
            'seconds_until': int(time_until.total_seconds())
        }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'next_run': None,
            'time_until': None
        }

def extract_schedule_from_workflow(file_path):
    """Extract schedule information from workflow file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if not content.strip():
            return {'error': 'Empty workflow file'}
            
        # Parse YAML
        try:
            workflow = yaml.safe_load(content)
        except yaml.YAMLError as e:
            return {'error': f'Invalid YAML format: {e}'}
            
        if not workflow:
            return {'error': 'Empty YAML content'}
            
        # Check for schedule in 'on' section
        if 'on' not in workflow:
            return {'error': 'No "on" section found'}
            
        on_section = workflow['on']
        schedules = None
        
        # Handle different formats
        if isinstance(on_section, dict):
            if 'schedule' in on_section:
                schedules = on_section['schedule']
            else:
                return {'error': 'No schedule section found in on dict'}
        elif isinstance(on_section, list):
            # Find schedule in list
            for item in on_section:
                if isinstance(item, dict) and 'schedule' in item:
                    schedules = item['schedule']
                    break
            if schedules is None:
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
        valid_schedules.sort(key=lambda x: x['seconds_until'])
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
        
        # Check for BotCore integration patterns
        botcore_patterns = [
            'BotCore',
            'workflow_data_integration.py',
            'Intelligence/data/integrated',
            'TradeSignalData',
            'RiskAssessment',
            'NewsSentiment'
        ]
        
        content_lower = content.lower()
        for pattern in botcore_patterns:
            if pattern.lower() in content_lower:
                return True
        
        return False
        
    except Exception:
        return False

def check_api_health(file_path):
    """Check what APIs the workflow uses"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        apis_found = []
        
        # Check for various API patterns
        if 'yfinance' in content or 'yf.Ticker' in content:
            apis_found.append('Yahoo Finance')
        
        if 'feedparser' in content or 'rss' in content.lower():
            apis_found.append('RSS Feeds')
        
        if 'requests.get' in content:
            apis_found.append('HTTP Requests')
        
        if 'api.' in content.lower():
            apis_found.append('Generic API')
            
        return apis_found
        
    except Exception:
        return []

def main():
    """Main verification function"""
    print("🔍 FIXED COMPREHENSIVE VERIFICATION: ALL 27 WORKFLOWS")
    print("=" * 80)
    print(f"Verification Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    workflows_dir = ".github/workflows"
    
    if not os.path.exists(workflows_dir):
        print(f"❌ ERROR: Workflows directory not found: {workflows_dir}")
        return
    
    workflow_files = [f for f in os.listdir(workflows_dir) if f.endswith('.yml')]
    workflow_files.sort()
    
    print(f"📊 FOUND {len(workflow_files)} WORKFLOW FILES\n")
    
    results = []
    scheduled_workflows = 0
    integrated_workflows = 0
    
    for i, workflow_file in enumerate(workflow_files, 1):
        file_path = os.path.join(workflows_dir, workflow_file)
        
        print(f"🔍 [{i:2d}/{len(workflow_files)}] Checking: {workflow_file}")
        
        # Extract schedule
        schedule_info = extract_schedule_from_workflow(file_path)
        has_schedule = 'error' not in schedule_info
        
        if has_schedule:
            scheduled_workflows += 1
        
        # Check BotCore integration
        has_integration = check_botcore_integration(file_path)
        if has_integration:
            integrated_workflows += 1
        
        # Check APIs
        apis = check_api_health(file_path)
        
        result = {
            'file': workflow_file,
            'has_schedule': has_schedule,
            'has_integration': has_integration,
            'schedule_info': schedule_info,
            'apis': apis
        }
        results.append(result)
        
        # Print immediate status
        schedule_status = "✅" if result['has_schedule'] else "❌"
        integration_status = "✅" if result['has_integration'] else "❌"
        api_count = len(apis)
        
        print(f"    Schedule: {schedule_status} | BotCore: {integration_status} | APIs: {api_count}")
        
        if result['has_schedule']:
            print(f"    Next Run: {schedule_info['next_run']}")
            print(f"    Time Until: {schedule_info['time_until']}")
            print(f"    Schedules: {schedule_info['total_schedules']} total, {schedule_info['valid_schedules']} valid")
        else:
            print(f"    Error: {schedule_info.get('error', 'Unknown error')}")
        
        if apis:
            print(f"    APIs: {', '.join(apis)}")
        
        print()
    
    # Summary statistics
    print("=" * 80)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 80)
    
    total_workflows = len(workflow_files)
    fully_complete = len([r for r in results if r['has_schedule'] and r['has_integration']])
    
    print(f"Total Workflows Found: {total_workflows}")
    print(f"With Valid Scheduling: {scheduled_workflows}/{total_workflows}")
    print(f"With BotCore Integration: {integrated_workflows}/{total_workflows}")
    print(f"Fully Complete (Schedule + Integration): {fully_complete}/{total_workflows}")
    
    # Show next run times
    print("\n⏰ NEXT RUN TIMES FOR SCHEDULED WORKFLOWS")
    print("=" * 80)
    
    scheduled_results = [r for r in results if r['has_schedule']]
    if scheduled_results:
        # Sort by next run time
        scheduled_results.sort(key=lambda x: x['schedule_info']['seconds_until'])
        
        for i, result in enumerate(scheduled_results[:10], 1):  # Show next 10
            name = result['file'].replace('.yml', '')
            next_run = result['schedule_info']['next_run']
            time_until = result['schedule_info']['time_until']
            print(f"{i:2d}. {name:<35} → {next_run} (in {time_until})")
    else:
        print("No scheduled workflows found!")
    
    # Show problematic workflows
    problematic = [r for r in results if not r['has_schedule'] or not r['has_integration']]
    
    if problematic:
        print("\n⚠️  PROBLEMATIC WORKFLOWS")
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
    
    # Save detailed results
    import json
    results_data = {
        'verification_time': datetime.now().isoformat(),
        'summary': {
            'total_workflows': total_workflows,
            'scheduled_workflows': scheduled_workflows,
            'integrated_workflows': integrated_workflows,
            'fully_complete': fully_complete
        },
        'detailed_results': results
    }
    
    with open('workflow_verification_results_fixed.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"💾 Detailed results saved to: workflow_verification_results_fixed.json")

if __name__ == "__main__":
    main()