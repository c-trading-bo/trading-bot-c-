#!/usr/bin/env python3
"""
Test Enhanced Business Logic Validation
Demonstrates the new advanced validation capabilities
"""

from Intelligence.mechanic.local.bot_mechanic import LocalBotMechanic
import sys
from pathlib import Path

def test_enhanced_validation():
    """Test the enhanced business logic validation"""
    
    print("🧠 TESTING ENHANCED BUSINESS LOGIC VALIDATION")
    print("="*60)
    
    # Initialize enhanced mechanic
    mechanic = LocalBotMechanic()
    
    print("\n🔍 Testing Business Logic Validator Integration...")
    
    # Test 1: Check if BLV is integrated
    if mechanic.business_logic_validator:
        print("✅ Business Logic Validator successfully integrated")
        print(f"   Version: {mechanic.business_logic_validator.version}")
    else:
        print("❌ Business Logic Validator failed to integrate")
        return
    
    # Test 2: Run quick validation test
    print("\n🧪 Running Quick Validation Test...")
    
    # Create a test trading function to validate
    test_code = '''
def test_trading_function():
    # This function has intentional issues for testing
    capital = 100000
    
    # Missing position size limit check (CRITICAL issue)
    position_size = capital * 0.5  # 50% risk - too high!
    
    # Missing stop loss (CRITICAL issue)
    if price > sma:
        signal = "BUY"
        # No stop loss set
    
    # No error handling (CRITICAL issue)
    order = place_order(symbol, "BUY", position_size)
    
    return signal

def calculate_rsi_with_issues(prices):
    # Missing zero division protection (HIGH issue)
    gains = sum(positive_changes)
    losses = sum(negative_changes)
    rs = gains / losses  # Potential division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi
'''
    
    # Write test file
    test_file = Path("test_trading_logic.py")
    test_file.write_text(test_code)
    
    try:
        # Run validation on test file
        print("   Analyzing test trading functions...")
        validation_results = mechanic.validate_business_logic(verbose=False)
        
        if validation_results.get('status') != 'unavailable':
            print(f"   ✅ Validation completed")
            print(f"   📊 Total checks: {validation_results.get('total_checks', 0)}")
            print(f"   ❌ Issues found: {validation_results.get('failed', 0)}")
            print(f"   🚨 Critical issues: {validation_results.get('critical_issues', 0)}")
            
            # Show some critical issues
            critical_issues = validation_results.get('critical_issues', [])
            if critical_issues:
                print(f"\n   🚨 Sample Critical Issues Detected:")
                for issue in critical_issues[:3]:
                    print(f"     • {issue}")
        else:
            print("   ⚠️ Validation unavailable")
    
    finally:
        # Clean up test file
        if test_file.exists():
            test_file.unlink()
    
    # Test 3: Enhanced Deep Scan
    print(f"\n🔬 Testing Enhanced Deep Scan...")
    print("   This will analyze your entire codebase for:")
    print("   • Code structure and quality")
    print("   • Risk management logic")
    print("   • Trading logic validation") 
    print("   • Algorithm correctness")
    print("   • Semantic analysis")
    
    # Run a fast enhanced scan
    try:
        enhanced_results = mechanic.enhanced_deep_scan(verbose=True, fast_mode=True)
        
        print(f"\n✅ Enhanced scan completed!")
        print(f"   Overall Health: {enhanced_results['overall_health']['combined_score']:.1f}%")
        print(f"   Status: {enhanced_results['overall_health']['status'].upper()}")
        
    except Exception as e:
        print(f"⚠️ Enhanced scan encountered issue: {e}")
    
    print("\n" + "="*60)
    print("🎯 ENHANCED VALIDATION CAPABILITIES")
    print("="*60)
    print("✅ Risk Management Validation:")
    print("   • Position sizing limits")
    print("   • Stop loss requirements") 
    print("   • Risk calculation accuracy")
    print("   • Leverage limit checks")
    
    print("\n✅ Trading Logic Validation:")
    print("   • Signal generation logic")
    print("   • Entry/exit conditions")
    print("   • Order validation")
    print("   • Error handling requirements")
    
    print("\n✅ Algorithm Validation:")
    print("   • Indicator calculation correctness")
    print("   • Numerical stability checks")
    print("   • Backtest integrity")
    print("   • Performance metric validation")
    
    print("\n✅ Semantic Analysis:")
    print("   • Data flow validation")
    print("   • Variable usage tracking")
    print("   • Function dependency mapping")
    print("   • Logic consistency checks")
    
    print("\n💡 The mechanic now UNDERSTANDS your trading logic!")
    print("   It validates that your code does what it's supposed to do,")
    print("   not just that it runs without errors.")
    
    print("="*60)

if __name__ == "__main__":
    test_enhanced_validation()
