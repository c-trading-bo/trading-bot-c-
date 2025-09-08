#!/usr/bin/env python3
"""
AUTO-START MECHANIC - Integrated with Bot Launch
Automatically starts Local Bot Mechanic when bot launches
"""

import subprocess
import sys
import threading
import time
from pathlib import Path

def start_mechanic_background():
    """Start Local Bot Mechanic in background"""
    try:
        mechanic_path = Path("Intelligence/mechanic/local/bot_mechanic.py")
        
        if not mechanic_path.exists():
            print("❌ Bot Mechanic not found")
            return None
        
        print("🧠 Starting Local Bot Mechanic background service...")
        
        # Start mechanic as background process
        process = subprocess.Popen([
            sys.executable, 
            str(mechanic_path)
        ], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True
        )
        
        print("✅ Local Bot Mechanic started in background")
        print(f"   Process ID: {process.pid}")
        
        return process
        
    except Exception as e:
        print(f"❌ Failed to start Local Bot Mechanic: {e}")
        return None

def get_mechanic_status():
    """Get current mechanic status for dashboard"""
    try:
        import json
        status_file = Path("Intelligence/mechanic/database/knowledge.json")
        
        if status_file.exists():
            with open(status_file, 'r') as f:
                data = json.load(f)
                
            last_scan = data.get('last_scan', {})
            
            return {
                'status': 'healthy' if len(last_scan.get('issues_found', [])) == 0 else 'warning',
                'health_score': 100 - (len(last_scan.get('issues_found', [])) * 10),
                'issues_count': len(last_scan.get('issues_found', [])),
                'files_tracked': len(data.get('files', {})),
                'last_scan_time': last_scan.get('timestamp', 'Never'),
                'auto_fixed': len(last_scan.get('auto_fixed', [])),
                'monitoring': True
            }
    except Exception:
        pass
    
    return {
        'status': 'starting',
        'health_score': 0,
        'issues_count': 0,
        'files_tracked': 0,
        'last_scan_time': 'Starting...',
        'auto_fixed': 0,
        'monitoring': False
    }

if __name__ == "__main__":
    # Start mechanic when called directly
    process = start_mechanic_background()
    
    if process:
        try:
            # Keep monitoring
            while True:
                time.sleep(10)
                if process and process.poll() is not None:
                    print("⚠️ Bot Mechanic process ended, restarting...")
                    process = start_mechanic_background()
        except KeyboardInterrupt:
            print("\n🛑 Stopping Bot Mechanic...")
            if process:
                process.terminate()
