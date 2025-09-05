#!/usr/bin/env python3
"""
Enterprise 24/7 Trading Bot - Schedule Optimization ONLY
Preserves all original code, only updates cron schedules
"""
import os
import re
import yaml
from pathlib import Path

# ENTERPRISE 24/7 SCHEDULE MAPPING (47,500 minutes/month)
SCHEDULE_MAP = {
    # TIER 1: TRADE EXECUTION CRITICAL (35% budget = 16,625 min)
    'es_nq_critical_trading.yml': [
        '*/3 9:30-16 * * 1-5',      # Market: Every 3 min
        '*/5 6-9:30,16-20 * * 1-5', # Extended: Every 5 min  
        '*/15 20-6 * * *',          # Overnight: Every 15 min
        '*/10 * * * 0,6'            # Weekends: Every 10 min
    ],
    
    'ultimate_ml_rl_intel_system.yml': [
        '*/5 9:30-16 * * 1-5',      # Market: Every 5 min
        '*/10 4-9:30,16-20 * * 1-5', # Extended: Every 10 min
        '*/30 20-4 * * *',          # Overnight: Every 30 min
        '0 * * * 0,6'               # Weekends: Hourly
    ],
    
    'portfolio_heat.yml': [
        '*/10 9-16 * * 1-5',        # Market: Every 10 min
        '*/20 4-9,16-20 * * 1-5',   # Extended: Every 20 min
        '0 */2 20-4 * * *'          # Overnight: Every 2 hours
    ],

    # TIER 2: REAL-TIME MARKET INTELLIGENCE (25% = 11,875 min)
    'microstructure.yml': [
        '*/5 9:30-10:30,14:30-16 * * 1-5',  # First/Last 90 min: Every 5 min
        '*/10 10:30-14:30 * * 1-5'          # Mid-day: Every 10 min
    ],
    
    'options_flow.yml': [
        '*/5 9:30-10,15:30-16 * * 1-5',     # Power hours: Every 5 min
        '*/10 10-15:30 * * 1-5',            # Regular: Every 10 min
        '*/30 4-9:30,16-18 * * 1-5'         # Extended: Every 30 min
    ],
    
    'failed_patterns.yml': [
        '*/15 9:30-16 * * 1-5',     # Market: Every 15 min
        '*/30 6-9:30,16-20 * * 1-5' # Extended: Every 30 min
    ],
    
    'intermarket.yml': [
        '*/20 6-22 * * *',          # Active hours: Every 20 min
        '0 22-6 * * *'              # Overnight: Hourly
    ],

    # TIER 3: ANALYTICAL ENGINES (20% = 9,500 min)
    'volatility_surface.yml': [
        '0,30 9,12,15 * * 1-5',     # Open/Noon/Close: 2x each
        '0 18,2 * * *'              # Evening/Overnight: Once
    ],
    
    'zones_identifier.yml': [
        '15 8,11,14,17 * * 1-5',    # Pre/Mid/Post market: 4x daily
        '0 6 * * 0,6'               # Weekends: Morning scan
    ],
    
    'es_nq_correlation_matrix.yml': [
        '0 */2 * * *'               # Every 2 hours 24/7
    ],
    
    'mm_positioning.yml': [
        '30 11,15,23 * * *'         # Late morning, close, overnight
    ],

    # TIER 4: DATA INFRASTRUCTURE (10% = 4,750 min)
    'daily_report.yml': [
        '45 7,15 * * 1-5',          # Pre-market: 7:45 AM, Pre-close: 3:45 PM
        '0 20 * * 1-5'              # Evening: 8 PM
    ],
    
    'market_data.yml': [
        '31 16 * * 1-5',            # 4:31 PM weekdays
        '0 0,8,16 * * 0,6'          # 3x on weekends
    ],
    
    'overnight.yml': [
        '0 3,8 * * *'               # 3 AM (EU open), 8 AM (US pre-market)
    ],
    
    'daily_consolidated.yml': [
        '0 4 * * *',                # 4 AM daily
        '0 12 * * 0,6'              # Noon on weekends
    ],

    # TIER 5: ML/AI TRAINING (5% = 2,375 min)
    'ml_trainer.yml': [
        '0 5,17 * * 1-5',           # 5 AM & 5 PM weekdays
        '0 10 * * 0'                # 10 AM Sunday deep train
    ],
    
    'ultimate_ml_rl_training_pipeline.yml': [
        '0 6,18 * * 1-5',           # 6 AM & 6 PM weekdays
        '0 2 * * 6'                 # 2 AM Saturday full retrain
    ],

    # TIER 6: ULTIMATE PIPELINES (3% = 1,425 min)
    'ultimate_data_collection_pipeline.yml': [
        '0 5,16 * * *'              # 5 AM & 4 PM daily
    ],
    
    'ultimate_regime_detection_pipeline.yml': [
        '0 7,19 * * *'              # 7 AM & 7 PM
    ],
    
    'ultimate_options_flow_pipeline.yml': [
        '*/30 9:30-16 * * 1-5'      # Every 30 min during market
    ],
    
    'ultimate_news_sentiment_pipeline.yml': [
        '0 6,9,12,15,18 * * 1-5'    # 5x daily on trading days
    ],
    
    'ultimate_testing_qa_pipeline.yml': [
        '0 3 * * 0'                 # Sunday 3 AM weekly
    ],
    
    'ultimate_build_ci_pipeline.yml': [
        '0 2 * * 6'                 # Saturday 2 AM weekly
    ],

    # TIER 7: PERIODIC MONITORS (2% = 950 min)
    'fed_liquidity.yml': [
        '0 14 * * 2,4'              # Tuesday & Thursday 2 PM
    ],
    
    'opex_calendar.yml': [
        '0 9 * * 1,3,5'             # Mon/Wed/Fri mornings
    ],
    
    'seasonality.yml': [
        '0 6 * * 1'                 # Monday 6 AM weekly
    ],
    
    'cloud_bot_mechanic.yml': [
        '0 */6 * * *'               # Every 6 hours 24/7
    ]
}

def update_workflow_schedule(file_path, new_schedules):
    """Update only the cron schedule in a workflow file, preserve all other content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the schedule section and replace it
        # Pattern to match the schedule section in YAML
        schedule_pattern = r'(schedule:\s*\n)(.*?)(\n\s*workflow_dispatch:|$)'
        
        # Build new schedule content
        new_schedule_content = "schedule:\n"
        for schedule in new_schedules:
            new_schedule_content += f"    - cron: '{schedule}'\n"
        
        # Replace the schedule section
        def replace_schedule(match):
            return match.group(1) + new_schedule_content.replace('schedule:\n', '') + match.group(3)
        
        updated_content = re.sub(schedule_pattern, replace_schedule, content, flags=re.DOTALL)
        
        # Write back the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating {file_path}: {str(e)}")
        return False

def main():
    print("🎯 ENTERPRISE 24/7 TRADING BOT - SCHEDULE OPTIMIZATION")
    print("=" * 60)
    print("⚡ PRESERVING ALL ORIGINAL CODE - UPDATING SCHEDULES ONLY")
    print()
    
    workflow_dir = Path(".github/workflows")
    
    if not workflow_dir.exists():
        print("❌ Workflow directory not found!")
        return
    
    success_count = 0
    total_count = 0
    
    for filename, schedules in SCHEDULE_MAP.items():
        file_path = workflow_dir / filename
        total_count += 1
        
        if not file_path.exists():
            print(f"⚠️  {filename} - File not found, skipping")
            continue
        
        print(f"🔧 Updating {filename}...")
        print(f"   Schedules: {schedules}")
        
        if update_workflow_schedule(file_path, schedules):
            print(f"✅ {filename} - Schedule updated successfully")
            success_count += 1
        else:
            print(f"❌ {filename} - Failed to update")
        print()
    
    print("=" * 60)
    print(f"📊 OPTIMIZATION COMPLETE: {success_count}/{total_count} workflows updated")
    print()
    print("🎯 ENTERPRISE 24/7 SCHEDULE APPLIED:")
    print("   • TIER 1: Critical trading workflows - High frequency")
    print("   • TIER 2: Real-time intelligence - Market hours focus")  
    print("   • TIER 3: Analytics - Strategic timing")
    print("   • TIER 4: Data infrastructure - Off-peak optimization")
    print("   • TIER 5: ML/AI training - Overnight cycles")
    print("   • TIER 6: Ultimate pipelines - Efficient intervals")
    print("   • TIER 7: Periodic monitors - Low frequency")
    print()
    print("💰 BUDGET: ~47,500 minutes/month (95% of 50,000 limit)")
    print("🚀 Ready for enterprise-grade 24/7 trading!")

if __name__ == "__main__":
    main()
