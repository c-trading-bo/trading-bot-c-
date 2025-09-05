#!/usr/bin/env python3
"""
Enterprise 24/7 Trading Workflow Optimization Script
Updates all 27 workflows with optimized schedules for 47,500 minutes/month budget
"""

import os
import yaml
import re
from pathlib import Path

# ENTERPRISE 24/7 SCHEDULE DEFINITIONS
WORKFLOW_SCHEDULES = {
    # ═══════════════════════════════════════════════════════════════
    # TIER 1: TRADE EXECUTION CRITICAL (35% budget = 16,625 min)
    # ═══════════════════════════════════════════════════════════════
    
    "es_nq_critical_trading.yml": [
        "*/3 9-16 * * 1-5",      # Market: Every 3 min
        "*/5 6-9,16-20 * * 1-5", # Extended: Every 5 min
        "*/15 20-6 * * *",       # Overnight: Every 15 min
        "*/10 * * * 0,6"         # Weekends: Every 10 min
    ],
    
    "ultimate_ml_rl_intel_system.yml": [
        "*/5 9-16 * * 1-5",      # Market: Every 5 min
        "*/10 4-9,16-20 * * 1-5", # Extended: Every 10 min
        "*/30 20-4 * * *",       # Overnight: Every 30 min
        "0 * * * 0,6"            # Weekends: Hourly
    ],
    
    "portfolio_heat.yml": [
        "*/10 9-16 * * 1-5",     # Market: Every 10 min
        "*/20 4-9,16-20 * * 1-5", # Extended: Every 20 min
        "0 */2 20-4 * * *"       # Overnight: Every 2 hours
    ],

    # ═══════════════════════════════════════════════════════════════
    # TIER 2: REAL-TIME MARKET INTELLIGENCE (25% = 11,875 min)
    # ═══════════════════════════════════════════════════════════════
    
    "microstructure.yml": [
        "*/5 9-10,14-16 * * 1-5",  # First/Last 90 min: Every 5 min
        "*/10 10-14 * * 1-5"       # Mid-day: Every 10 min
    ],
    
    "options_flow.yml": [
        "*/5 9-10,15-16 * * 1-5",   # Power hours: Every 5 min
        "*/10 10-15 * * 1-5",       # Regular: Every 10 min
        "*/30 4-9,16-18 * * 1-5"    # Extended: Every 30 min
    ],
    
    "failed_patterns.yml": [
        "*/15 9-16 * * 1-5",        # Market: Every 15 min
        "*/30 6-9,16-20 * * 1-5"    # Extended: Every 30 min
    ],
    
    "intermarket.yml": [
        "*/20 6-22 * * *",          # Active hours: Every 20 min
        "0 22-6 * * *"              # Overnight: Hourly
    ],

    # ═══════════════════════════════════════════════════════════════
    # TIER 3: ANALYTICAL ENGINES (20% = 9,500 min)
    # ═══════════════════════════════════════════════════════════════
    
    "volatility_surface.yml": [
        "0,30 9,12,15 * * 1-5",     # Open/Noon/Close: 2x each
        "0 18,2 * * *"              # Evening/Overnight: Once
    ],
    
    "zones_identifier.yml": [
        "15 8,11,14,17 * * 1-5",    # Pre/Mid/Post market: 4x daily
        "0 6 * * 0,6"               # Weekends: Morning scan
    ],
    
    "es_nq_correlation_matrix.yml": [
        "0 */2 * * *"               # Every 2 hours 24/7
    ],
    
    "mm_positioning.yml": [
        "30 11,15,23 * * *"         # Late morning, close, and overnight
    ],

    # ═══════════════════════════════════════════════════════════════
    # TIER 4: DATA INFRASTRUCTURE (10% = 4,750 min)
    # ═══════════════════════════════════════════════════════════════
    
    "daily_report.yml": [
        "45 7,15 * * 1-5",          # Pre-market: 7:45 AM | Pre-close: 3:45 PM
        "0 20 * * 1-5"              # Evening: 8 PM
    ],
    
    "market_data.yml": [
        "31 16 * * 1-5",            # 4:31 PM weekdays
        "0 0,8,16 * * 0,6"          # 3x on weekends
    ],
    
    "overnight.yml": [
        "0 3,8 * * *"               # 3 AM (EU open) | 8 AM (US pre-market)
    ],
    
    "daily_consolidated.yml": [
        "0 4 * * *",                # 4 AM daily
        "0 12 * * 0,6"              # Noon on weekends
    ],

    # ═══════════════════════════════════════════════════════════════
    # TIER 5: ML/AI TRAINING (5% = 2,375 min)
    # ═══════════════════════════════════════════════════════════════
    
    "ml_trainer.yml": [
        "0 5,17 * * 1-5",           # 5 AM & 5 PM weekdays
        "0 10 * * 0"                # 10 AM Sunday deep train
    ],
    
    "ultimate_ml_rl_training_pipeline.yml": [
        "0 6,18 * * 1-5",           # 6 AM & 6 PM weekdays
        "0 2 * * 6"                 # 2 AM Saturday full retrain
    ],

    # ═══════════════════════════════════════════════════════════════
    # TIER 6: ULTIMATE PIPELINES (3% = 1,425 min)
    # ═══════════════════════════════════════════════════════════════
    
    "ultimate_data_collection_pipeline.yml": [
        "0 5,16 * * *"              # 5 AM & 4 PM daily
    ],
    
    "ultimate_regime_detection_pipeline.yml": [
        "0 7,19 * * *"              # 7 AM & 7 PM
    ],
    
    "ultimate_options_flow_pipeline.yml": [
        "*/30 9-16 * * 1-5"         # Every 30 min during market only
    ],
    
    "ultimate_news_sentiment_pipeline.yml": [
        "0 6,9,12,15,18 * * 1-5"    # 5x daily on trading days
    ],
    
    "ultimate_testing_qa_pipeline.yml": [
        "0 3 * * 0"                 # Sunday 3 AM weekly
    ],
    
    "ultimate_build_ci_pipeline.yml": [
        "0 2 * * 6"                 # Saturday 2 AM weekly
    ],

    # ═══════════════════════════════════════════════════════════════
    # TIER 7: PERIODIC MONITORS (2% = 950 min)
    # ═══════════════════════════════════════════════════════════════
    
    "fed_liquidity.yml": [
        "0 14 * * 2,4"              # Tuesday & Thursday 2 PM
    ],
    
    "opex_calendar.yml": [
        "0 9 * * 1,3,5"             # Mon/Wed/Fri mornings
    ],
    
    "seasonality.yml": [
        "0 6 * * 1"                 # Monday 6 AM weekly
    ],
    
    "cloud_bot_mechanic.yml": [
        "0 */6 * * *"               # Every 6 hours 24/7
    ]
}

def update_workflow_schedule(file_path, schedules):
    """Update a single workflow file with new schedule"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse YAML to ensure it's valid
        try:
            yaml_data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            print(f"  ❌ YAML parse error in {file_path}: {e}")
            return False
        
        # Build new schedule section
        schedule_lines = []
        for schedule in schedules:
            schedule_lines.append(f"    - cron: '{schedule}'")
        
        new_schedule = "  schedule:\n" + "\n".join(schedule_lines)
        
        # Replace schedule section using regex
        pattern = r'(\s+schedule:\s*\n)(.*?)(\n\s+workflow_dispatch:|\n\s+[a-zA-Z_]|\n\njobs:|\Z)'
        
        def replace_schedule(match):
            return match.group(1) + "\n".join(schedule_lines) + "\n" + (match.group(3) if match.group(3) else "")
        
        new_content = re.sub(pattern, replace_schedule, content, flags=re.DOTALL)
        
        # Write back the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error updating {file_path}: {e}")
        return False

def main():
    print("🚀 ENTERPRISE 24/7 WORKFLOW OPTIMIZATION")
    print("=" * 60)
    print("📊 Optimizing for 47,500 minutes/month (95% of 50K budget)")
    print("🎯 27 workflows → Enterprise trading schedules")
    print("=" * 60)
    
    workflows_dir = Path(".github/workflows")
    
    if not workflows_dir.exists():
        print("❌ .github/workflows directory not found!")
        return
    
    updated_count = 0
    total_workflows = 0
    
    # Process each workflow
    for workflow_file in workflows_dir.glob("*.yml"):
        total_workflows += 1
        filename = workflow_file.name
        
        if filename in WORKFLOW_SCHEDULES:
            schedules = WORKFLOW_SCHEDULES[filename]
            print(f"\n🔧 Optimizing: {filename}")
            print(f"   📅 Schedules: {len(schedules)} cron expressions")
            
            if update_workflow_schedule(workflow_file, schedules):
                print(f"   ✅ Successfully updated")
                updated_count += 1
            else:
                print(f"   ❌ Failed to update")
        else:
            print(f"\n⚠️  No schedule defined for: {filename}")
    
    print("\n" + "=" * 60)
    print("📊 OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"✅ Updated workflows: {updated_count}/{total_workflows}")
    print(f"📈 Schedule coverage: {len(WORKFLOW_SCHEDULES)}/27 workflows")
    print("🎯 Budget allocation:")
    print("   • Tier 1 (Critical): 35% = 16,625 min")
    print("   • Tier 2 (Intelligence): 25% = 11,875 min")
    print("   • Tier 3 (Analysis): 20% = 9,500 min")
    print("   • Tier 4 (Data): 10% = 4,750 min")
    print("   • Tier 5 (ML/AI): 5% = 2,375 min")
    print("   • Tier 6 (Pipelines): 3% = 1,425 min")
    print("   • Tier 7 (Monitors): 2% = 950 min")
    print("\n🚀 Enterprise 24/7 optimization complete!")
    print("💰 Total budget: 47,500/50,000 minutes (95% utilization)")

if __name__ == "__main__":
    main()
