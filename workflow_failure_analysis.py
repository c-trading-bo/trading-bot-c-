#!/usr/bin/env python3
"""
Workflow Failure Analysis Tool
Analyzes workflow failures and provides debugging information
"""

import os
import json
from datetime import datetime

def analyze_workflow_issues():
    print("🔍 WORKFLOW FAILURE ANALYSIS")
    print("=" * 50)
    print()
    
    print("📋 COMMON WORKFLOW FAILURE CAUSES:")
    print()
    
    print("1️⃣ DEPENDENCY ISSUES:")
    print("   • Missing Python packages")
    print("   • Version conflicts")
    print("   • Environment setup failures")
    print()
    
    print("2️⃣ AUTHENTICATION ISSUES:")
    print("   • Expired secrets")
    print("   • Missing API keys")
    print("   • Permission problems")
    print()
    
    print("3️⃣ RESOURCE LIMITS:")
    print("   • GitHub Actions timeout (6 hours max)")
    print("   • Memory/CPU limits")
    print("   • Network connectivity")
    print()
    
    print("4️⃣ CODE ISSUES:")
    print("   • Syntax errors")
    print("   • Import failures")
    print("   • Logic errors")
    print()
    
    print("🔧 DEBUGGING STEPS:")
    print()
    print("1. Check specific workflow logs:")
    print("   https://github.com/c-trading-bo/trading-bot-c-/actions")
    print()
    print("2. Look for error patterns:")
    print("   • Red X marks ❌")
    print("   • Error messages in logs")
    print("   • Step that failed")
    print()
    print("3. Common fixes:")
    print("   • Update dependencies")
    print("   • Fix syntax errors")
    print("   • Add missing secrets")
    print("   • Reduce resource usage")
    print()
    
    # Check for local workflow files
    workflow_dir = ".github/workflows"
    if os.path.exists(workflow_dir):
        print("📁 LOCAL WORKFLOW FILES FOUND:")
        workflow_files = [f for f in os.listdir(workflow_dir) if f.endswith('.yml')]
        print(f"   Found {len(workflow_files)} workflow files")
        
        # Show key workflows
        key_workflows = [
            'es_nq_critical_trading.yml',
            'ultimate_ml_rl_intel_system.yml',
            'es_nq_correlation_matrix.yml'
        ]
        
        print("\n🎯 KEY WORKFLOWS TO CHECK:")
        for workflow in key_workflows:
            if workflow in workflow_files:
                print(f"   ✅ {workflow}")
            else:
                print(f"   ❌ {workflow} - MISSING!")
    
    print("\n" + "=" * 50)
    print("📊 NEXT STEPS:")
    print("1. Visit: https://github.com/c-trading-bo/trading-bot-c-/actions")
    print("2. Click on failed workflow run (red X)")
    print("3. Expand failed step to see error")
    print("4. Share error message for specific help")
    print()
    print("🎯 What specific error are you seeing?")

def create_workflow_health_summary():
    """Create a summary of workflow health without API access"""
    current_time = datetime.now()
    
    print(f"\n🕐 CURRENT TIME: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 DAY: {current_time.strftime('%A')}")
    
    hour = current_time.hour
    day = current_time.strftime('%A')
    
    if day == 'Saturday':
        print("🚫 SATURDAY: Futures closed - workflows should be idle")
        expected_runs = "None (Saturday futures closure)"
    elif day == 'Sunday':
        print("📅 SUNDAY: Minimal activity")
        expected_runs = "Every 30min (ES/NQ) and 2hr (ML/RL)"
    elif 9 <= hour < 16:
        print("🚀 MARKET HOURS: High frequency trading")
        expected_runs = "Every 3min (ES/NQ) and 5min (ML/RL)"
    elif 7 <= hour < 9 or 16 <= hour < 18:
        print("⏰ PRE/POST MARKET: Medium frequency")
        expected_runs = "Every 5min (ES/NQ) and 10min (ML/RL)"
    else:
        print("🌙 OVERNIGHT: Low frequency monitoring")
        expected_runs = "Every 15min (ES/NQ) and 30min (ML/RL)"
    
    print(f"🎯 EXPECTED RUNS: {expected_runs}")
    print("\n💡 If workflows are failing, check the Actions tab for specific errors!")

if __name__ == "__main__":
    analyze_workflow_issues()
    create_workflow_health_summary()
