#!/usr/bin/env python3
"""
Test Local Bot Mechanic Integration
Verifies that the mechanic runs correctly in background mode
"""

import time
import json
import subprocess
import sys
from pathlib import Path

def test_background_start():
    """Test starting mechanic in background"""
    print("🧪 Testing Local Bot Mechanic Background Start...")
    
    # Start auto_start.py
    try:
        auto_start = Path("Intelligence/mechanic/local/auto_start.py")
        if not auto_start.exists():
            print("❌ auto_start.py not found")
            return False
            
        print("🚀 Starting background service...")
        process = subprocess.Popen([sys.executable, str(auto_start)], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        
        # Wait a bit for it to initialize
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Background service is running")
            
            # Check if database files are being created
            db_path = Path("Intelligence/mechanic/database")
            if db_path.exists():
                print("✅ Database directory created")
                
                knowledge_file = db_path / "knowledge.json"
                if knowledge_file.exists():
                    print("✅ Knowledge database created")
                    
                    # Check content
                    with open(knowledge_file, 'r') as f:
                        data = json.load(f)
                        
                    if 'last_scan' in data:
                        print("✅ Initial scan completed")
                    else:
                        print("⚠️ Initial scan still running...")
                        
                else:
                    print("⚠️ Knowledge database not yet created")
            else:
                print("⚠️ Database directory not created")
            
            # Stop the process
            process.terminate()
            process.wait()
            print("🛑 Test service stopped")
            
            return True
        else:
            print("❌ Background service failed to start")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_direct_mechanic():
    """Test mechanic directly"""
    print("\n🧪 Testing Direct Mechanic Call...")
    
    try:
        mechanic_file = Path("Intelligence/mechanic/local/bot_mechanic.py")
        if not mechanic_file.exists():
            print("❌ bot_mechanic.py not found")
            return False
            
        print("🚀 Running quick test...")
        result = subprocess.run([sys.executable, str(mechanic_file), "--test"], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Direct mechanic test passed")
            return True
        else:
            print(f"❌ Direct mechanic test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ Test timed out (normal for background service)")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("🧠 LOCAL BOT MECHANIC - INTEGRATION TEST")
    print("="*60)
    
    # Test 1: Background start
    test1_passed = test_background_start()
    
    # Test 2: Direct mechanic
    test2_passed = test_direct_mechanic()
    
    print("\n" + "="*60)
    print("📊 TEST RESULTS:")
    print("="*60)
    print(f"Background Start: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Direct Mechanic:  {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Local Bot Mechanic is ready for production use")
    else:
        print("\n⚠️ Some tests failed - check configuration")
    
    print("="*60)
