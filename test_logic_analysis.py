#!/usr/bin/env python3
"""Test script to demonstrate code logic analysis"""

from Intelligence.mechanic.local.bot_mechanic import LocalBotMechanic
import ast

def test_logic_analysis():
    mech = LocalBotMechanic()
    
    print("=== CODE LOGIC ANALYSIS CAPABILITIES ===\n")
    
    # Test function analysis
    test_code = '''
def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    try:
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        return 100 - (100 / (1 + avg_gain / avg_loss))
    except Exception as e:
        return None

def place_order(symbol, side, qty):
    # Missing error handling - bad!
    return api.place_order(symbol, side, qty)
'''
    
    tree = ast.parse(test_code)
    
    print("🔍 ANALYZING TRADING FUNCTIONS:\n")
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            analysis = mech._analyze_function(node, test_code)
            print(f"Function: {analysis['name']}")
            print(f"  ✅ Category: {analysis['category']}")
            print(f"  🛡️ Error Handling: {'✅ YES' if analysis['has_error_handling'] else '❌ NO'}")
            print(f"  📝 Documentation: {'✅ YES' if analysis['has_docstring'] else '❌ NO'}")
            print(f"  📋 Arguments: {analysis['args']}")
            print(f"  🌐 API Calls: {'✅ YES' if analysis['calls_apis'] else '❌ NO'}")
            print()
    
    print("🧠 WHAT THE MECHANIC KNOWS:")
    print("  ✅ Detects missing error handling")
    print("  ✅ Identifies trading logic patterns")
    print("  ✅ Validates function signatures")
    print("  ✅ Checks for API integration")
    print("  ✅ Ensures proper documentation")
    print("  ✅ Categorizes function purposes")

if __name__ == "__main__":
    test_logic_analysis()
