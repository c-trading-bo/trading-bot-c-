#!/usr/bin/env python3
"""
Test script for Local Bot Mechanic
"""

import sys
import os
from pathlib import Path

# Add mechanic to path
mechanic_path = Path("Intelligence/mechanic/local")
sys.path.insert(0, str(mechanic_path))

def test_imports():
    """Test basic imports"""
    print("🧪 Testing imports...")
    
    try:
        # Import the main file as a module
        import importlib.util
        spec = importlib.util.spec_from_file_location("bot_mechanic", 
                                                     mechanic_path / "bot_mechanic.py")
        bot_mechanic = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bot_mechanic)
        
        print("✅ bot_mechanic module loaded")
        
        # Test class access
        LocalBotMechanic = bot_mechanic.LocalBotMechanic
        print("✅ LocalBotMechanic class found")
        
        return LocalBotMechanic
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return None

def test_mechanic(LocalBotMechanic):
    """Test mechanic functionality"""
    print("\n🧪 Testing LocalBotMechanic...")
    
    try:
        # Create instance
        mechanic = LocalBotMechanic()
        print(f"✅ Created mechanic v{mechanic.version}")
        
        # Test quick scan
        result = mechanic.quick_scan()
        print(f"✅ Quick scan completed: {result['healthy']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mechanic test failed: {e}")
        return False

def test_launcher():
    """Test the auto-launcher"""
    print("\n🧪 Testing auto-launcher...")
    
    try:
        # Test launcher import
        import importlib.util
        spec = importlib.util.spec_from_file_location("start_local_mechanic", 
                                                     mechanic_path / "start_local_mechanic.py")
        launcher_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher_module)
        
        print("✅ Launcher module loaded")
        
        # Test launcher class
        MechanicAutoLauncher = launcher_module.MechanicAutoLauncher
        launcher = MechanicAutoLauncher()
        print(f"✅ Launcher created v{launcher.version}")
        
        return True
        
    except Exception as e:
        print(f"❌ Launcher test failed: {e}")
        return False

def main():
    """Main test function"""
    print("="*60)
    print("🧠 LOCAL BOT MECHANIC - SYSTEM TEST")
    print("="*60)
    
    # Test imports
    LocalBotMechanic = test_imports()
    if not LocalBotMechanic:
        print("❌ Failed at import stage")
        return False
    
    # Test mechanic
    if not test_mechanic(LocalBotMechanic):
        print("❌ Failed at mechanic stage")
        return False
    
    # Test launcher
    if not test_launcher():
        print("❌ Failed at launcher stage")
        return False
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("🚀 Local Bot Mechanic system is ready to use!")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
