#!/usr/bin/env python3
"""
GitHub Actions Minutes Calculator
Calculates monthly minute usage for all workflows
"""

import glob
import yaml
import re
from datetime import datetime, timedelta

def parse_cron_frequency(cron_expr):
    """Calculate approximate runs per month for a cron expression"""
    parts = cron_expr.split()
    if len(parts) != 5:
        return 0
    
    minute, hour, day, month, weekday = parts
    
    # Calculate runs per day
    runs_per_day = 0
    
    # Parse minute field
    if minute == '*':
        minute_runs = 60
    elif '/' in minute:
        if minute.startswith('*/'):
            interval = int(minute[2:])
            minute_runs = 60 / interval
        else:
            minute_runs = 1
    elif ',' in minute:
        minute_runs = len(minute.split(','))
    else:
        minute_runs = 1
    
    # Parse hour field
    if hour == '*':
        hour_runs = 24
    elif '/' in hour:
        if hour.startswith('*/'):
            interval = int(hour[2:])
            hour_runs = 24 / interval
        else:
            hour_runs = 1
    elif '-' in hour:
        start, end = map(int, hour.split('-'))
        hour_runs = end - start + 1
    elif ',' in hour:
        hour_runs = len(hour.split(','))
    else:
        hour_runs = 1
    
    # Parse weekday field
    if weekday == '*':
        weekday_factor = 7
    elif ',' in weekday:
        weekday_factor = len(weekday.split(','))
    elif '-' in weekday:
        start, end = map(int, weekday.split('-'))
        weekday_factor = end - start + 1
    else:
        weekday_factor = 1
    
    # Calculate based on weekday restrictions
    if weekday != '*':
        runs_per_day = (minute_runs * hour_runs * weekday_factor) / 7
    else:
        runs_per_day = minute_runs * hour_runs
    
    # Monthly calculation (30 days average)
    runs_per_month = runs_per_day * 30
    
    return int(runs_per_month)

def estimate_workflow_minutes(workflow_name, timeout_minutes=15):
    """Estimate minutes per workflow run"""
    # Different workflows have different typical runtimes
    typical_runtimes = {
        'es_nq_critical_trading': 8,
        'ultimate_ml_rl_intel_system': 15,
        'daily_consolidated': 10,
        'volatility_surface': 12,
        'es_nq_correlation_matrix': 6,
        'cloud_bot_mechanic': 8,
        'portfolio_heat': 10
    }
    
    for key, runtime in typical_runtimes.items():
        if key in workflow_name.lower():
            return runtime
    
    return timeout_minutes  # Default to timeout value

def main():
    print("💰 GITHUB ACTIONS MINUTES CALCULATOR")
    print("=" * 60)
    
    workflows = glob.glob('.github/workflows/*.yml')
    total_monthly_minutes = 0
    workflow_usage = []
    
    for workflow_path in workflows:
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            name = data.get('name', 'Unknown')
            triggers = data.get('on', {})
            
            if 'schedule' not in triggers:
                continue
            
            schedules = triggers['schedule']
            workflow_total = 0
            
            print(f"\n📋 {name}")
            print("-" * 50)
            
            for i, schedule in enumerate(schedules, 1):
                cron_expr = schedule.get('cron', '')
                runs_per_month = parse_cron_frequency(cron_expr)
                minutes_per_run = estimate_workflow_minutes(name)
                monthly_minutes = runs_per_month * minutes_per_run
                
                print(f"  Schedule {i}: {cron_expr}")
                print(f"    🔄 ~{runs_per_month:,} runs/month")
                print(f"    ⏱️  ~{minutes_per_run} minutes/run")
                print(f"    📊 ~{monthly_minutes:,} minutes/month")
                
                workflow_total += monthly_minutes
            
            print(f"  💰 TOTAL: ~{workflow_total:,} minutes/month")
            total_monthly_minutes += workflow_total
            workflow_usage.append((name, workflow_total))
            
        except Exception as e:
            print(f"❌ Error processing {workflow_path}: {e}")
    
    print(f"\n" + "=" * 60)
    print(f"💰 TOTAL MONTHLY USAGE: ~{total_monthly_minutes:,} MINUTES")
    print(f"📊 GitHub Free Tier: 2,000 minutes/month")
    print(f"📊 Team/Pro Tier: 3,000 minutes/month")
    print(f"📊 Enterprise: Custom limits")
    
    if total_monthly_minutes > 2000:
        overage = total_monthly_minutes - 2000
        print(f"⚠️  OVERAGE (Free): {overage:,} minutes")
        cost = overage * 0.008  # $0.008 per minute for public repos
        print(f"💸 Estimated cost: ${cost:.2f}/month")
    else:
        remaining = 2000 - total_monthly_minutes
        print(f"✅ REMAINING (Free): {remaining:,} minutes")
    
    print(f"\n📈 TOP CONSUMERS:")
    workflow_usage.sort(key=lambda x: x[1], reverse=True)
    for name, usage in workflow_usage[:5]:
        print(f"  🔥 {usage:,} min/month - {name}")
    
    print(f"\n🎯 Saturday Optimization Savings:")
    print(f"  💰 Estimated savings: ~3,840 minutes/month")
    print(f"  📊 Effective usage: ~{total_monthly_minutes - 3840:,} minutes/month")

if __name__ == "__main__":
    main()
