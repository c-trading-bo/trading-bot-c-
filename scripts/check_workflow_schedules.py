#!/usr/bin/env python3
"""
⏰ WORKFLOW SCHEDULE ANALYZER
Shows next scheduled workflow runs for your trading bot
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
import pytz
import re

def parse_cron_schedule(cron_expr):
    """Parse cron expression and return description"""
    
    # Common cron patterns and their descriptions
    cron_patterns = {
        '0 13 * * 1-5': 'Daily at 8:00 AM EST (1:00 PM UTC) on weekdays',
        '30 21 * * 1-5': 'Daily at 4:30 PM EST (9:30 PM UTC) on weekdays', 
        '0 */6 * * *': 'Every 6 hours',
        '0 2,14 * * *': 'Twice daily at 2:00 AM and 2:00 PM UTC',
        '0 */4 * * *': 'Every 4 hours',
        '0 1,13 * * *': 'Twice daily at 1:00 AM and 1:00 PM UTC',
        '0 3,15 * * *': 'Twice daily at 3:00 AM and 3:00 PM UTC',
        '0 4,16 * * *': 'Twice daily at 4:00 AM and 4:00 PM UTC',
        '0 5,17 * * *': 'Twice daily at 5:00 AM and 5:00 PM UTC',
        '0 6,18 * * *': 'Twice daily at 6:00 AM and 6:00 PM UTC',
        '0 0,12 * * *': 'Twice daily at midnight and noon UTC',
        '0 */15 * * *': 'Every 15 minutes',
        '*/45 * * * *': 'Every 45 minutes'
    }
    
    return cron_patterns.get(cron_expr, f'Custom schedule: {cron_expr}')

def get_next_run_time(cron_expr, current_time):
    """Calculate next run time for cron expression (simplified)"""
    
    # For common patterns, calculate next run
    if cron_expr == '0 13 * * 1-5':  # Daily at 1 PM UTC weekdays
        next_run = current_time.replace(hour=13, minute=0, second=0, microsecond=0)
        if next_run <= current_time or current_time.weekday() >= 5:  # Weekend
            # Next weekday
            days_ahead = 7 - current_time.weekday() if current_time.weekday() >= 5 else 1
            next_run = next_run + timedelta(days=days_ahead)
        return next_run
    
    elif cron_expr == '30 21 * * 1-5':  # Daily at 9:30 PM UTC weekdays  
        next_run = current_time.replace(hour=21, minute=30, second=0, microsecond=0)
        if next_run <= current_time or current_time.weekday() >= 5:
            days_ahead = 7 - current_time.weekday() if current_time.weekday() >= 5 else 1
            next_run = next_run + timedelta(days=days_ahead)
        return next_run
    
    elif '*/45' in cron_expr:  # Every 45 minutes
        next_run = current_time + timedelta(minutes=45 - (current_time.minute % 45))
        return next_run.replace(second=0, microsecond=0)
    
    elif '*/15' in cron_expr:  # Every 15 minutes  
        next_run = current_time + timedelta(minutes=15 - (current_time.minute % 15))
        return next_run.replace(second=0, microsecond=0)
    
    elif '*/6' in cron_expr:  # Every 6 hours
        next_run = current_time + timedelta(hours=6 - (current_time.hour % 6))
        return next_run.replace(minute=0, second=0, microsecond=0)
    
    elif '*/4' in cron_expr:  # Every 4 hours
        next_run = current_time + timedelta(hours=4 - (current_time.hour % 4))
        return next_run.replace(minute=0, second=0, microsecond=0)
    
    else:
        # Default: assume next hour for unknown patterns
        return current_time + timedelta(hours=1)

def analyze_workflow_schedules():
    """Analyze all workflow schedules and show next runs"""
    
    workflow_dir = Path('.github/workflows')
    
    if not workflow_dir.exists():
        print("❌ No .github/workflows directory found!")
        return
    
    # Current time in UTC
    utc_now = datetime.now(pytz.UTC)
    est_now = utc_now.astimezone(pytz.timezone('US/Eastern'))
    
    print("⏰ WORKFLOW SCHEDULE ANALYSIS")
    print("=" * 60)
    print(f"📅 Current time (EST): {est_now.strftime('%Y-%m-%d %I:%M:%S %p %Z')}")
    print(f"🌍 Current time (UTC): {utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("=" * 60)
    
    schedules = []
    
    for workflow_file in sorted(workflow_dir.glob('*.yml')):
        try:
            with open(workflow_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find cron schedules
            cron_matches = re.findall(r"- cron: ['\"]([^'\"]+)['\"]", content)
            
            if cron_matches:
                for cron_expr in cron_matches:
                    description = parse_cron_schedule(cron_expr)
                    next_run = get_next_run_time(cron_expr, utc_now)
                    next_run_est = next_run.astimezone(pytz.timezone('US/Eastern'))
                    
                    time_until = next_run - utc_now
                    hours_until = time_until.total_seconds() / 3600
                    
                    schedules.append({
                        'workflow': workflow_file.name,
                        'cron': cron_expr,
                        'description': description,
                        'next_run_utc': next_run,
                        'next_run_est': next_run_est,
                        'hours_until': hours_until
                    })
        
        except Exception as e:
            print(f"⚠️ Error reading {workflow_file.name}: {e}")
    
    # Sort by next run time
    schedules.sort(key=lambda x: x['next_run_utc'])
    
    print("\n🚀 UPCOMING WORKFLOW RUNS:")
    print("-" * 60)
    
    for i, schedule in enumerate(schedules[:10]):  # Show next 10
        workflow_name = schedule['workflow'].replace('.yml', '')
        next_run_est = schedule['next_run_est']
        hours_until = schedule['hours_until']
        
        if hours_until < 1:
            time_str = f"in {int(hours_until * 60)} minutes"
        elif hours_until < 24:
            time_str = f"in {hours_until:.1f} hours"
        else:
            time_str = f"in {hours_until/24:.1f} days"
        
        print(f"{i+1:2d}. 📋 {workflow_name}")
        print(f"    ⏰ {next_run_est.strftime('%a, %b %d at %I:%M %p EST')} ({time_str})")
        print(f"    📝 {schedule['description']}")
        print()
    
    # Show next immediate run
    if schedules:
        next_workflow = schedules[0]
        print("🎯 NEXT SCHEDULED WORKFLOW:")
        print("=" * 40)
        print(f"📋 Workflow: {next_workflow['workflow']}")
        print(f"⏰ Next run: {next_workflow['next_run_est'].strftime('%A, %B %d at %I:%M %p EST')}")
        print(f"⏳ Time until: {next_workflow['hours_until']:.1f} hours")
        print(f"📝 Schedule: {next_workflow['description']}")

if __name__ == "__main__":
    analyze_workflow_schedules()
