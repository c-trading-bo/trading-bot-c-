#!/usr/bin/env python3
"""
Ultimate ML/RL System Health Monitor
Continuous monitoring of all system components
"""

import time
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

class SystemHealthMonitor:
    def __init__(self):
        self.components = {
            'data_collection': {
                'market_data': 'Intelligence/data/market/latest.json',
                'news_data': 'Intelligence/data/news/latest.json', 
                'zones_data': 'Intelligence/data/zones/active_zones.json',
                'regime_data': 'Intelligence/data/regime/current.json'
            },
            'models': {
                'neural_bandits': 'Intelligence/models/bandits/neural_bandit.onnx',
                'regime_detector': 'Intelligence/models/regime/random_forest.pkl'
            },
            'workflows': {
                'ultimate_system': '.github/workflows/ultimate_ml_rl_intel_system.yml',
                'continuous_training': '.github/workflows/train-continuous-final.yml',
                'news_pulse': '.github/workflows/news_pulse.yml'
            }
        }
        
    def check_file_freshness(self, file_path, max_age_hours=24):
        """Check if file exists and is fresh"""
        if not os.path.exists(file_path):
            return {'status': 'MISSING', 'age_hours': None}
        
        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        age = datetime.now() - file_time
        age_hours = age.total_seconds() / 3600
        
        if age_hours > max_age_hours:
            return {'status': 'STALE', 'age_hours': age_hours}
        else:
            return {'status': 'FRESH', 'age_hours': age_hours}
    
    def check_data_quality(self, file_path):
        """Check data file quality and content"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Basic quality checks
            has_timestamp = 'timestamp' in data
            is_recent = False
            
            if has_timestamp:
                timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                age = datetime.now() - timestamp.replace(tzinfo=None)
                is_recent = age.total_seconds() < 86400  # Less than 24 hours
            
            return {
                'valid_json': True,
                'has_timestamp': has_timestamp,
                'is_recent': is_recent,
                'data_size': len(str(data))
            }
        except Exception as e:
            return {
                'valid_json': False,
                'error': str(e)
            }
    
    def check_workflow_activity(self):
        """Check GitHub Actions workflow activity"""
        # In a real environment, this would check GitHub API
        # For now, we'll check workflow file modifications
        workflow_activity = {}
        
        for name, path in self.components['workflows'].items():
            if os.path.exists(path):
                freshness = self.check_file_freshness(path, max_age_hours=168)  # 1 week
                workflow_activity[name] = {
                    'file_exists': True,
                    'last_modified': freshness
                }
            else:
                workflow_activity[name] = {
                    'file_exists': False
                }
        
        return workflow_activity
    
    def generate_health_report(self):
        """Generate comprehensive health report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'HEALTHY',
            'components': {},
            'alerts': [],
            'recommendations': []
        }
        
        # Check all data components
        for category, files in self.components.items():
            report['components'][category] = {}
            
            for component, file_path in files.items():
                if category == 'data_collection':
                    freshness = self.check_file_freshness(file_path, max_age_hours=6)
                    quality = self.check_data_quality(file_path) if freshness['status'] != 'MISSING' else {}
                    
                    report['components'][category][component] = {
                        'freshness': freshness,
                        'quality': quality
                    }
                    
                    # Generate alerts
                    if freshness['status'] == 'MISSING':
                        report['alerts'].append(f"⚠️  {component} data file missing")
                        report['overall_status'] = 'DEGRADED'
                    elif freshness['status'] == 'STALE':
                        report['alerts'].append(f"⚠️  {component} data is stale ({freshness['age_hours']:.1f}h old)")
                        report['overall_status'] = 'DEGRADED'
                    
                elif category == 'models':
                    exists = os.path.exists(file_path)
                    freshness = self.check_file_freshness(file_path, max_age_hours=168) if exists else None
                    
                    report['components'][category][component] = {
                        'exists': exists,
                        'freshness': freshness
                    }
                    
                    if not exists:
                        report['alerts'].append(f"❌ {component} model missing")
                        report['overall_status'] = 'CRITICAL'
        
        # Check workflow activity
        report['workflow_activity'] = self.check_workflow_activity()
        
        # Generate recommendations
        if len(report['alerts']) == 0:
            report['recommendations'].append("✅ All systems operational - continue monitoring")
        else:
            if any('missing' in alert.lower() for alert in report['alerts']):
                report['recommendations'].append("🔧 Run data collection workflows manually")
            if any('stale' in alert.lower() for alert in report['alerts']):
                report['recommendations'].append("⏱️  Check workflow scheduling and execution")
            if any('model' in alert.lower() for alert in report['alerts']):
                report['recommendations'].append("🧠 Trigger model training workflows")
        
        return report
    
    def save_health_report(self, report):
        """Save health report to file"""
        os.makedirs("Intelligence/reports/health", exist_ok=True)
        
        # Save latest report
        with open("Intelligence/reports/health/latest.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Archive report with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_path = f"Intelligence/reports/health/health_{timestamp}.json"
        with open(archive_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return archive_path
    
    def print_health_summary(self, report):
        """Print human-readable health summary"""
        print(f"\n🏥 SYSTEM HEALTH REPORT - {report['timestamp']}")
        print("=" * 60)
        print(f"Overall Status: {report['overall_status']}")
        
        # Data components
        print(f"\n📊 Data Components:")
        for component, data in report['components']['data_collection'].items():
            freshness = data['freshness']
            status_icon = "✅" if freshness['status'] == 'FRESH' else "⚠️ " if freshness['status'] == 'STALE' else "❌"
            age_text = f" ({freshness['age_hours']:.1f}h old)" if freshness['age_hours'] else " (missing)"
            print(f"  {status_icon} {component}{age_text}")
        
        # Models
        print(f"\n🧠 ML Models:")
        for component, data in report['components']['models'].items():
            status_icon = "✅" if data['exists'] else "❌"
            print(f"  {status_icon} {component}")
        
        # Workflows
        print(f"\n⚙️  Workflows:")
        for workflow, data in report['workflow_activity'].items():
            status_icon = "✅" if data['file_exists'] else "❌"
            print(f"  {status_icon} {workflow}")
        
        # Alerts
        if report['alerts']:
            print(f"\n🚨 ALERTS ({len(report['alerts'])}):")
            for alert in report['alerts']:
                print(f"  {alert}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"  {rec}")
        
        print("=" * 60)
    
    def monitor_continuous(self, interval_minutes=30):
        """Run continuous monitoring"""
        print(f"🔄 Starting continuous monitoring (every {interval_minutes} minutes)")
        
        while True:
            try:
                report = self.generate_health_report()
                archive_path = self.save_health_report(report)
                self.print_health_summary(report)
                
                print(f"\n💾 Report saved: {archive_path}")
                print(f"⏰ Next check in {interval_minutes} minutes...")
                
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                print("\n🛑 Monitoring stopped by user")
                break
            except Exception as e:
                print(f"\n❌ Monitoring error: {e}")
                time.sleep(60)  # Wait 1 minute before retry

def main():
    monitor = SystemHealthMonitor()
    
    # Generate single report
    report = monitor.generate_health_report()
    archive_path = monitor.save_health_report(report)
    monitor.print_health_summary(report)
    
    print(f"\n💾 Health report saved to: {archive_path}")
    print(f"📊 Latest report: Intelligence/reports/health/latest.json")

if __name__ == "__main__":
    main()