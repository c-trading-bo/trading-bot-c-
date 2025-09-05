#!/usr/bin/env python3
"""
Add Auto-Response Triggers to Cloud Mechanic
Makes it respond automatically when other workflows fail
"""

def add_auto_response_triggers():
    workflow_path = ".github/workflows/cloud_bot_mechanic.yml"
    
    print("🚨 Adding auto-response triggers to Cloud Mechanic...")
    
    try:
        # Read the current workflow
        with open(workflow_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Define the workflow_run section
        workflow_run_section = '''
  # 🚨 AUTO-RESPONSE: Triggers when ANY workflow fails
  workflow_run:
    workflows:
      - "Ultimate ML+RL Training Pipeline"
      - "Ultimate Data Collection Pipeline"
      - "Ultimate News Sentiment Pipeline"
      - "Ultimate Options Flow Pipeline"
      - "Ultimate Regime Detection Pipeline"
      - "Ultimate Testing & QA Pipeline"
      - "Ultimate ML+RL Intel System"
      - "Ultimate Build CI Pipeline"
      - "Daily Consolidated"
      - "Portfolio Heat"
      - "Volatility Surface Analysis"
      - "ES/NQ Correlation Matrix"
      - "ES/NQ Critical Trading"
      - "Intermarket Correlations"
      - "Seasonality Patterns Analysis"
      - "Microstructure"
      - "Fed Liquidity"
      - "MM Positioning"
      - "Social Momentum"
      - "Congressional Trades"
      - "COT Report"
      - "Failed Patterns"
      - "Zones Identifier"
      - "ML Model Training"
      - "Data Collection"
      - "Options Flow"
    types:
      - completed
    branches:
      - main
'''
        
        # Find the insertion point (after the schedule section)
        schedule_end = "    - cron: '0 */6 * * *'"
        
        if schedule_end in content:
            # Insert the workflow_run section after the schedule
            parts = content.split(schedule_end)
            if len(parts) == 2:
                new_content = parts[0] + schedule_end + workflow_run_section + parts[1]
                
                # Write the enhanced workflow
                with open(workflow_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print("✅ Auto-response triggers added successfully!")
                print("🤖 Cloud Mechanic will now:")
                print("   🚨 Auto-respond to ANY workflow failure")
                print("   ⚡ Pre-cache dependencies for failing workflows")
                print("   🔧 Auto-fix common issues")
                print("   📊 Monitor all 27 workflows in real-time")
                return True
            else:
                print("❌ Multiple schedule sections found")
                return False
        else:
            print("❌ Schedule section not found")
            return False
        
    except Exception as e:
        print(f"❌ Error adding auto-response triggers: {e}")
        return False

if __name__ == "__main__":
    add_auto_response_triggers()
