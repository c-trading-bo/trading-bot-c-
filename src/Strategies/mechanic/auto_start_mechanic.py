#!/usr/bin/env python3
"""
AUTO-START LOCAL BOT MECHANIC
Automatically starts with bot launch and integrates with main dashboard
NO SEPARATE DASHBOARD - integrated into main bot dashboard
"""

import sys
import os
import json
import time
import threading
import subprocess
from datetime import datetime
from pathlib import Path

# Add the local mechanic to path
mechanic_path = Path(__file__).parent / "local"
sys.path.insert(0, str(mechanic_path))

try:
    from bot_mechanic import LocalBotMechanic
except ImportError:
    print("❌ Bot Mechanic not found. Please ensure it's properly installed.")
    sys.exit(1)

class AutoStartMechanic:
    def __init__(self):
        print("🚀 Auto-starting Local Bot Mechanic...")
        self.mechanic = LocalBotMechanic()
        self.running = True
        self.monitor_thread = None
        
    def start_background_monitoring(self):
        """Start background monitoring that integrates with main dashboard"""
        def monitor_loop():
            while self.running:
                try:
                    # Run quick scan every 60 seconds
                    results = self.mechanic.quick_scan()
                    
                    # Auto-fix issues if found
                    if not results['healthy']:
                        print(f"🔧 Bot Mechanic: Auto-fixing {results['issues']} issues...")
                        self.mechanic.auto_fix_all()
                    
                    # Save status for main dashboard
                    dashboard_data = self.mechanic.get_dashboard_data()
                    self._save_status_for_dashboard(dashboard_data)
                    
                    time.sleep(60)  # Check every minute
                    
                except Exception as e:
                    print(f"❌ Mechanic monitor error: {e}")
                    time.sleep(30)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("✅ Bot Mechanic background monitoring started")
        
    def _save_status_for_dashboard(self, data):
        """Save mechanic status for main dashboard integration"""
        status_dir = Path("Intelligence/mechanic/database")
        status_dir.mkdir(parents=True, exist_ok=True)
        
        status_file = status_dir / "dashboard_status.json"
        with open(status_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def get_current_status(self):
        """Get current status for main dashboard"""
        return self.mechanic.get_dashboard_data()
    
    def run_full_scan(self):
        """Run full scan - called by main dashboard"""
        return self.mechanic.deep_scan(verbose=False)
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        print("🛑 Bot Mechanic monitoring stopped")

# Global instance for main bot to use
auto_mechanic = None

def start_auto_mechanic():
    """Start the auto mechanic - called by main bot"""
    global auto_mechanic
    if auto_mechanic is None:
        auto_mechanic = AutoStartMechanic()
        auto_mechanic.start_background_monitoring()
        
        # Run initial scan
        print("🔍 Running initial bot scan...")
        auto_mechanic.mechanic.deep_scan(verbose=True)
        
    return auto_mechanic

def get_mechanic_status():
    """Get mechanic status for main dashboard"""
    global auto_mechanic
    if auto_mechanic is None:
        return {"status": "not_running", "healthy": False}
    return auto_mechanic.get_current_status()

def stop_auto_mechanic():
    """Stop the auto mechanic"""
    global auto_mechanic
    if auto_mechanic:
        auto_mechanic.stop()
        auto_mechanic = None

if __name__ == "__main__":
    # For testing only
    mechanic = start_auto_mechanic()
    try:
        input("Press Enter to stop...")
    except KeyboardInterrupt:
        pass
    finally:
        stop_auto_mechanic()
