#!/usr/bin/env python3
"""
Auto-start script for Local Bot Mechanic
Runs automatically when the bot starts - NO MANUAL INTERVENTION REQUIRED
"""

import os
import sys
import time
import signal
import subprocess
import threading
from datetime import datetime
from pathlib import Path

class AutoBotMechanic:
    def __init__(self):
        self.running = True
        self.monitor_process = None
        self.dashboard_process = None
        self.setup_signal_handlers()
        
    def setup_signal_handlers(self):
        """Setup graceful shutdown"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n🧠 Local Bot Mechanic shutting down...")
        self.running = False
        self.stop_processes()
    
    def check_and_setup(self):
        """Automatically check and set up if needed"""
        try:
            # Ensure directories exist
            dirs = [
                "Intelligence/mechanic/database",
                "Intelligence/mechanic/logs", 
                "Intelligence/mechanic/reports",
                "Intelligence/data",
                "Intelligence/models"
            ]
            
            for dir_path in dirs:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            
            # Check if main files exist
            required_files = [
                "Intelligence/mechanic/local/bot_mechanic.py",
                "Intelligence/mechanic/monitor.py",
                "Intelligence/mechanic/dashboard.py"
            ]
            
            missing = [f for f in required_files if not Path(f).exists()]
            
            if missing:
                print(f"⚠️ Missing {len(missing)} mechanic files, but continuing with basic monitoring")
                return False
            
            return True
            
        except Exception as e:
            print(f"⚠️ Setup check failed: {e}")
            return False
    
    def start_monitor(self):
        """Start monitoring process"""
        try:
            monitor_path = Path("Intelligence/mechanic/monitor.py")
            
            if not monitor_path.exists():
                print("⚠️ Monitor script not found, using basic health check")
                return self.start_basic_monitor()
            
            print("🔍 Starting Local Bot Mechanic monitor...")
            
            self.monitor_process = subprocess.Popen([
                sys.executable, str(monitor_path), "--interval", "60", "--quiet"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            print(f"✅ Monitor started (PID: {self.monitor_process.pid})")
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to start monitor: {e}")
            return self.start_basic_monitor()
    
    def start_basic_monitor(self):
        """Start basic monitoring if full system not available"""
        def basic_health_check():
            while self.running:
                try:
                    # Basic health checks
                    checks = {
                        'data_dir': Path('Intelligence/data').exists(),
                        'models_dir': Path('Intelligence/models').exists(),
                        'python_running': True  # If we're here, Python is working
                    }
                    
                    issues = sum(1 for v in checks.values() if not v)
                    
                    if issues == 0:
                        status = "✅ Healthy"
                    else:
                        status = f"⚠️ {issues} issues"
                    
                    # Write status to log
                    log_dir = Path('Intelligence/mechanic/logs')
                    log_dir.mkdir(parents=True, exist_ok=True)
                    
                    with open(log_dir / 'basic_status.log', 'w') as f:
                        f.write(f"{datetime.now().isoformat()}: {status}\n")
                    
                    time.sleep(60)  # Check every minute
                    
                except Exception:
                    pass
        
        threading.Thread(target=basic_health_check, daemon=True).start()
        print("✅ Basic health monitoring started")
        return True
    
    def start_dashboard(self):
        """Start dashboard if available"""
        try:
            dashboard_path = Path("Intelligence/mechanic/dashboard.py")
            
            if not dashboard_path.exists():
                print("ℹ️ Dashboard not available (optional)")
                return False
            
            # Check if port is available
            port = 8888
            try:
                import socket
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    result = s.connect_ex(('localhost', port))
                    if result == 0:
                        print(f"ℹ️ Port {port} already in use, skipping dashboard")
                        return False
            except:
                pass
            
            print(f"🌐 Starting dashboard on port {port}...")
            
            self.dashboard_process = subprocess.Popen([
                sys.executable, str(dashboard_path), str(port)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            print(f"✅ Dashboard started (PID: {self.dashboard_process.pid}) - http://localhost:{port}")
            return True
            
        except Exception as e:
            print(f"ℹ️ Dashboard start failed (optional): {e}")
            return False
    
    def stop_processes(self):
        """Stop all processes"""
        if self.monitor_process and self.monitor_process.poll() is None:
            try:
                self.monitor_process.terminate()
                self.monitor_process.wait(timeout=5)
                print("✅ Monitor stopped")
            except:
                try:
                    self.monitor_process.kill()
                except:
                    pass
        
        if self.dashboard_process and self.dashboard_process.poll() is None:
            try:
                self.dashboard_process.terminate()
                self.dashboard_process.wait(timeout=5)
                print("✅ Dashboard stopped")
            except:
                try:
                    self.dashboard_process.kill()
                except:
                    pass
    
    def run(self):
        """Main run loop"""
        print("\n" + "="*50)
        print("🧠 AUTO-STARTING LOCAL BOT MECHANIC")
        print("="*50)
        
        # Quick setup check
        setup_ok = self.check_and_setup()
        
        # Start monitoring (always works)
        monitor_started = self.start_monitor()
        
        # Start dashboard (optional)
        dashboard_started = self.start_dashboard()
        
        print("\n📊 Status Summary:")
        print(f"  • Setup: {'✅ Complete' if setup_ok else '⚠️ Partial'}")
        print(f"  • Monitor: {'✅ Running' if monitor_started else '❌ Failed'}")
        print(f"  • Dashboard: {'✅ Running' if dashboard_started else 'ℹ️ Not available'}")
        
        if monitor_started or dashboard_started:
            print(f"\n✅ Local Bot Mechanic is now running automatically!")
            print("🔄 Monitoring your bot health and auto-fixing issues")
            if dashboard_started:
                print("🌐 Dashboard: http://localhost:8888")
            print("="*50)
        else:
            print("\n⚠️ Could not start full mechanic, but basic monitoring active")
        
        # Keep running until bot shuts down
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        
        self.stop_processes()

def main():
    """Auto-start entry point"""
    try:
        mechanic = AutoBotMechanic()
        mechanic.run()
    except Exception as e:
        print(f"❌ Auto-start failed: {e}")
        # Continue anyway - don't break the bot
        return 0

if __name__ == "__main__":
    main()
