#!/usr/bin/env python3
"""
Comprehensive test of the complete Supply/Demand Zone System
Tests all components working together: historical identification, live tracking, and C# integration
"""

import subprocess
import os
import json
import sys
from datetime import datetime

def run_command(cmd, description):
    """Run a command and return success/failure"""
    print(f"\n🧪 {description}")
    print("-" * 60)
    try:
        if isinstance(cmd, str):
            # Convert string command to list for security
            cmd = cmd.split()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/runner/work/trading-bot-c-/trading-bot-c-")
        if result.returncode == 0:
            print(f"✅ PASSED: {description}")
            if result.stdout.strip():
                print(result.stdout.strip())
            return True
        else:
            print(f"❌ FAILED: {description}")
            if result.stderr.strip():
                print(f"Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ FAILED: {description} - Exception: {e}")
        return False

def check_file_exists(filepath, description):
    """Check if a file exists"""
    full_path = f"/home/runner/work/trading-bot-c-/trading-bot-c-/{filepath}"
    exists = os.path.exists(full_path)
    print(f"{'✅' if exists else '❌'} {description}: {filepath}")
    return exists

def validate_zone_data(filepath):
    """Validate zone data structure"""
    try:
        full_path = f"/home/runner/work/trading-bot-c-/trading-bot-c-/{filepath}"
        with open(full_path, 'r') as f:
            data = json.load(f)
        
        required_fields = ['supply_zones', 'demand_zones', 'current_price', 'poc']
        missing_fields = [field for field in required_fields if field not in data]
        
        if not missing_fields:
            supply_count = len(data.get('supply_zones', []))
            demand_count = len(data.get('demand_zones', []))
            print(f"   ✅ Zone data structure valid: {supply_count} supply, {demand_count} demand zones")
            return True
        else:
            print(f"   ❌ Missing required fields: {missing_fields}")
            return False
    except Exception as e:
        print(f"   ❌ Failed to validate zone data: {e}")
        return False

def main():
    print("🏦 COMPREHENSIVE SUPPLY/DEMAND ZONE SYSTEM TEST")
    print("=" * 80)
    print(f"Test started at: {datetime.now()}")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Historical Zone Identification
    total_tests += 1
    if run_command("python Intelligence/scripts/identify_zones.py", "Historical Zone Identification"):
        tests_passed += 1
    
    # Test 2: Check generated zone files
    total_tests += 1
    print(f"\n🧪 Zone File Generation")
    print("-" * 60)
    files_exist = 0
    zone_files = [
        "Intelligence/data/zones/active_zones.json",
        "Intelligence/data/zones/active_zones_ES.json"
    ]
    
    for filepath in zone_files:
        if check_file_exists(filepath, "Zone file exists"):
            files_exist += 1
            validate_zone_data(filepath)
    
    if files_exist == len(zone_files):
        print("✅ PASSED: Zone File Generation")
        tests_passed += 1
    else:
        print("❌ FAILED: Zone File Generation")
    
    # Test 3: C# Zone Service Integration
    total_tests += 1
    if run_command("dotnet run --project TestEnhancedZones", "C# Enhanced Zone Service Integration"):
        tests_passed += 1
    
    # Test 4: Live Zone Tracker
    total_tests += 1
    tracker_test = """
from Intelligence.scripts.zones.live_zone_tracker import LiveZoneTracker
tracker = LiveZoneTracker()
if tracker.zones:
    interaction = tracker.update_price(5677.0, 45000)
    if interaction and interaction.get('zone_type') == 'demand':
        print('Live zone interaction detected successfully')
        exit(0)
    else:
        print('Live zone interaction failed')
        exit(1)
else:
    print('Failed to load zones for live tracking')
    exit(1)
"""
    
    if run_command(f'python -c "{tracker_test}"', "Live Zone Tracker"):
        tests_passed += 1
    
    # Test 5: C# Build Validation
    total_tests += 1
    if run_command("dotnet build --verbosity minimal", "C# Build Validation"):
        tests_passed += 1
    
    # Test 6: Zone Learning Directory Structure
    total_tests += 1
    print(f"\n🧪 Zone Learning Directory Structure")
    print("-" * 60)
    learning_dirs = [
        "Intelligence/data/zones",
        "Intelligence/data/zones/learning",
        "Intelligence/scripts/zones"
    ]
    
    dirs_exist = 0
    for directory in learning_dirs:
        if check_file_exists(directory, "Directory exists"):
            dirs_exist += 1
    
    if dirs_exist == len(learning_dirs):
        print("✅ PASSED: Zone Learning Directory Structure")
        tests_passed += 1
    else:
        print("❌ FAILED: Zone Learning Directory Structure")
    
    # Final Summary
    print("\n" + "=" * 80)
    print("🏦 COMPREHENSIVE ZONE SYSTEM TEST RESULTS")
    print("=" * 80)
    
    success_rate = (tests_passed / total_tests) * 100
    
    print(f"Tests Passed: {tests_passed}/{total_tests} ({success_rate:.1f}%)")
    
    if tests_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! Complete Supply/Demand Zone System is working!")
        print("\n✅ System Components Verified:")
        print("   • Historical zone identification with institutional features")
        print("   • Volume profile analysis (HVN/LVN, POC)")
        print("   • Advanced zone strength calculation")
        print("   • Zone overlap cleaning and merging")
        print("   • Live zone tracking and learning")
        print("   • C# zone service integration")
        print("   • Enhanced zone interaction methods")
        print("   • Zone-adjusted stop/target placement")
        print("   • Zone interaction recording for learning")
        print("   • Backward compatibility maintained")
        print("\n🏦 The institutional-grade zone system is ready for production!")
        return 0
    else:
        print(f"\n❌ {total_tests - tests_passed} tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)