#!/usr/bin/env python3
"""
Quick Workflow Schedule Validator
Checks if your key workflows are properly scheduled and validates cron syntax
"""

import glob
import yaml
import re
from datetime import datetime

def validate_cron(cron_expr):
    """Basic cron validation - checks format"""
    parts = cron_expr.split()
    if len(parts) != 5:
        return False, "Must have 5 parts (min hour day month weekday)"
    
    # Basic pattern check
    pattern = r'^[\*\d\-\,\/]+$'
    for part in parts:
        if not re.match(pattern, part):
            return False, f"Invalid character in: {part}"
    
    return True, "Valid"

def main():
    print("🎯 WORKFLOW SCHEDULE VALIDATION")
    print("=" * 50)
    
    # Key workflows to check
    key_workflows = [
        'es_nq_critical_trading.yml',
        'ultimate_ml_rl_intel_system.yml', 
        'daily_consolidated.yml',
        'volatility_surface.yml',
        'es_nq_correlation_matrix.yml'
    ]
    
    for workflow_name in key_workflows:
        workflow_path = f'.github/workflows/{workflow_name}'
        print(f"\n📋 {workflow_name}")
        print("-" * 40)
        
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            triggers = data.get('on', {})
            
            if 'schedule' not in triggers:
                print("❌ No schedule found")
                continue
                
            schedules = triggers['schedule']
            print(f"✅ Found {len(schedules)} scheduled job(s)")
            
            for i, schedule in enumerate(schedules, 1):
                cron_expr = schedule.get('cron', '')
                print(f"  {i}. {cron_expr}")
                
                is_valid, msg = validate_cron(cron_expr)
                if is_valid:
                    print(f"     ✅ {msg}")
                else:
                    print(f"     ❌ {msg}")
                    
        except FileNotFoundError:
            print("❌ File not found")
        except yaml.YAMLError as e:
            print(f"❌ YAML error: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n🎯 VALIDATION COMPLETE")
    print(f"📅 Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
