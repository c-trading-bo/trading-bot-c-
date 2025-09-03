#!/usr/bin/env python3
"""
Monitor Script for Local Bot Mechanic
Lightweight continuous monitoring without full GUI
"""

import os
import sys
import time
import json
import signal
from datetime import datetime
from pathlib import Path

class LightweightMonitor:
    def __init__(self):
        self.running = True
        self.last_check = None
        self.issues_count = 0
        self.setup_signal_handlers()
        
    def setup_signal_handlers(self):
        """Setup graceful shutdown"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n\n👋 Shutting down monitor gracefully...")
        self.running = False
    
    def check_critical_systems(self):
        """Quick check of critical systems"""
        checks = {
            'python_files': self.check_python_files(),
            'data_freshness': self.check_data_freshness(),
            'model_availability': self.check_models(),
            'workflow_status': self.check_workflows(),
            'memory_usage': self.check_memory()
        }
        
        issues = sum(1 for result in checks.values() if not result['healthy'])
        return {
            'timestamp': datetime.now().isoformat(),
            'total_checks': len(checks),
            'issues_found': issues,
            'healthy': issues == 0,
            'details': checks
        }
    
    def check_python_files(self):
        """Check if critical Python files exist and are valid"""
        critical_files = [
            'Intelligence/scripts/strategies/es_nq_realtime.py',
            'Intelligence/scripts/strategies/spy_qqq_regime.py',
            'Intelligence/mechanic/local/bot_mechanic.py'
        ]
        
        missing = []
        for file_path in critical_files:
            if not Path(file_path).exists():
                missing.append(file_path)
        
        return {
            'healthy': len(missing) == 0,
            'missing_files': missing,
            'total_checked': len(critical_files)
        }
    
    def check_data_freshness(self):
        """Check if data files are reasonably fresh"""
        data_dir = Path('Intelligence/data')
        if not data_dir.exists():
            return {'healthy': False, 'reason': 'Data directory missing'}
        
        data_files = list(data_dir.glob('*.json')) + list(data_dir.glob('*.csv'))
        if not data_files:
            return {'healthy': False, 'reason': 'No data files found'}
        
        # Check if any file is newer than 24 hours
        now = time.time()
        fresh_files = [
            f for f in data_files 
            if (now - f.stat().st_mtime) < 86400  # 24 hours
        ]
        
        return {
            'healthy': len(fresh_files) > 0,
            'total_files': len(data_files),
            'fresh_files': len(fresh_files)
        }
    
    def check_models(self):
        """Check if ML models exist"""
        model_dir = Path('Intelligence/models')
        if not model_dir.exists():
            return {'healthy': False, 'reason': 'Models directory missing'}
        
        model_files = list(model_dir.glob('*.pkl')) + list(model_dir.glob('*.h5')) + list(model_dir.glob('*.joblib'))
        
        return {
            'healthy': len(model_files) > 0,
            'model_count': len(model_files)
        }
    
    def check_workflows(self):
        """Check GitHub workflows"""
        workflow_dir = Path('.github/workflows')
        if not workflow_dir.exists():
            return {'healthy': False, 'reason': 'No workflows directory'}
        
        yml_files = list(workflow_dir.glob('*.yml'))
        
        return {
            'healthy': len(yml_files) > 0,
            'workflow_count': len(yml_files)
        }
    
    def check_memory(self):
        """Check system memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            return {
                'healthy': memory.percent < 90,
                'usage_percent': memory.percent,
                'available_gb': memory.available / (1024**3)
            }
        except ImportError:
            return {
                'healthy': True,
                'reason': 'psutil not available'
            }
    
    def save_status(self, status):
        """Save status to file"""
        status_dir = Path('Intelligence/mechanic/logs')
        status_dir.mkdir(parents=True, exist_ok=True)
        
        status_file = status_dir / 'monitor_status.json'
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
    
    def print_status(self, status):
        """Print status to console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if status['healthy']:
            print(f"[{timestamp}] ✅ All systems healthy", end='\r')
        else:
            print(f"[{timestamp}] ⚠️  {status['issues_found']} issues detected")
            
            # Print details for issues
            for system, details in status['details'].items():
                if not details['healthy']:
                    reason = details.get('reason', 'Unknown issue')
                    print(f"  └─ {system}: {reason}")
    
    def auto_fix_attempt(self, status):
        """Attempt basic auto-fixes"""
        fixed_count = 0
        
        # Create missing directories
        dirs_to_create = [
            'Intelligence/data',
            'Intelligence/models',
            'Intelligence/scripts/strategies'
        ]
        
        for dir_path in dirs_to_create:
            path = Path(dir_path)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                fixed_count += 1
        
        return fixed_count
    
    def run_monitor(self, interval=60, auto_fix=True):
        """Run the monitoring loop"""
        print("\n" + "="*60)
        print("🔍 LOCAL BOT MECHANIC - LIGHTWEIGHT MONITOR")
        print("="*60)
        print(f"📊 Checking every {interval} seconds")
        print(f"🔧 Auto-fix: {'Enabled' if auto_fix else 'Disabled'}")
        print("📍 Press Ctrl+C to stop")
        print("="*60)
        
        consecutive_issues = 0
        
        while self.running:
            try:
                # Run health check
                status = self.check_critical_systems()
                self.last_check = status['timestamp']
                
                # Save status
                self.save_status(status)
                
                # Print status
                self.print_status(status)
                
                # Track consecutive issues
                if not status['healthy']:
                    consecutive_issues += 1
                    self.issues_count = status['issues_found']
                    
                    # Attempt auto-fix if enabled
                    if auto_fix and consecutive_issues <= 3:  # Don't spam fixes
                        fixed = self.auto_fix_attempt(status)
                        if fixed > 0:
                            print(f"  🔧 Auto-fixed {fixed} issues")
                    
                    # Alert if issues persist
                    if consecutive_issues >= 5:
                        print(f"\n⚠️  ALERT: {status['issues_found']} issues persist after {consecutive_issues} checks")
                        print("🔧 Consider running full Bot Mechanic scan")
                        
                else:
                    consecutive_issues = 0
                    self.issues_count = 0
                
                # Wait for next check
                time.sleep(interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ Monitor error: {e}")
                time.sleep(30)  # Wait before retrying
        
        print(f"\n📊 Final Status:")
        print(f"  • Last check: {self.last_check}")
        print(f"  • Issues found: {self.issues_count}")
        print("👋 Monitor stopped")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Local Bot Mechanic Lightweight Monitor')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds (default: 60)')
    parser.add_argument('--no-autofix', action='store_true', help='Disable automatic fixes')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    monitor = LightweightMonitor()
    
    try:
        monitor.run_monitor(
            interval=args.interval,
            auto_fix=not args.no_autofix
        )
    except Exception as e:
        print(f"❌ Monitor failed to start: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
