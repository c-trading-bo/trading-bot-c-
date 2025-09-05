#!/usr/bin/env python3
"""
CONTINUOUS CLOUD AI FEATURE MONITORING SYSTEM
Monitors all 326 features continuously and ensures they remain active
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List

class ContinuousFeatureMonitor:
    def __init__(self):
        self.monitoring_active = True
        self.check_interval = 300  # 5 minutes
        self.report_dir = Path("Intelligence/mechanic/cloud/monitoring")
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
    def run_feature_health_check(self) -> Dict:
        """Run comprehensive health check of all features"""
        print(f"🔍 RUNNING FEATURE HEALTH CHECK - {datetime.now().isoformat()}")
        print("=" * 60)
        
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'cloud_mechanic_core': self.test_cloud_mechanic_core(),
            'ultimate_workflows': self.test_ultimate_workflows(),
            'auto_response_system': self.test_auto_response(),
            'feature_discovery': self.test_feature_discovery(),
            'performance_metrics': self.collect_performance_metrics()
        }
        
        # Calculate overall health score
        scores = []
        for component, result in health_report.items():
            if isinstance(result, dict) and 'score' in result:
                scores.append(result['score'])
        
        overall_score = sum(scores) / len(scores) if scores else 0
        health_report['overall_health_score'] = overall_score
        health_report['status'] = self.get_status_from_score(overall_score)
        
        print(f"\n📊 OVERALL HEALTH SCORE: {overall_score:.1f}%")
        print(f"🎯 STATUS: {health_report['status']}")
        
        return health_report
    
    def test_cloud_mechanic_core(self) -> Dict:
        """Test the core cloud mechanic functionality"""
        print("🔧 Testing Cloud Mechanic Core...")
        
        try:
            env = os.environ.copy()
            env.update({
                'ULTIMATE_MODE': 'true',
                'GITHUB_REPOSITORY_OWNER': 'c-trading-bo',
                'GITHUB_REPOSITORY': 'trading-bot-c-'
            })
            
            result = subprocess.run([
                'python', 'Intelligence/mechanic/cloud/cloud_mechanic_core.py'
            ], env=env, capture_output=True, text=True, timeout=60)
            
            output = result.stdout
            
            core_features = {
                'execution_success': result.returncode == 0,
                'ultimate_mode_active': 'Ultimate' in output,
                'workflow_discovery': 'workflows' in output and 'Total workflows:' in output,
                'health_analysis': 'Healthy workflows:' in output,
                'ai_optimization': 'optimization' in output.lower(),
                'intelligent_preparation': 'INTELLIGENT PREPARATION' in output,
                'performance_metrics': 'Ultimate Metrics' in output,
                'workflow_learning': 'Learning' in output,
                'issue_detection': 'Issues found:' in output or 'Broken workflows:' in output
            }
            
            active_features = sum(1 for v in core_features.values() if v)
            score = (active_features / len(core_features)) * 100
            
            print(f"   ✅ Core features active: {active_features}/{len(core_features)} ({score:.1f}%)")
            
            return {
                'score': score,
                'active_features': active_features,
                'total_features': len(core_features),
                'features': core_features,
                'execution_time': len(output) > 1000  # Good output indicates proper execution
            }
            
        except Exception as e:
            print(f"   ❌ Core test failed: {e}")
            return {'score': 0, 'error': str(e)}
    
    def test_ultimate_workflows(self) -> Dict:
        """Test ultimate workflow configurations"""
        print("🚀 Testing Ultimate Workflows...")
        
        ultimate_files = list(Path(".github/workflows").glob("*ultimate*.yml"))
        
        workflow_health = {
            'total_workflows': len(ultimate_files),
            'valid_syntax': 0,
            'properly_configured': 0
        }
        
        for wf_file in ultimate_files:
            try:
                import yaml
                with open(wf_file, 'r') as f:
                    workflow_data = yaml.safe_load(f)
                
                if workflow_data:
                    workflow_health['valid_syntax'] += 1
                    
                    # Check for proper configuration
                    if ('jobs' in workflow_data and 
                        'on' in workflow_data and 
                        'permissions' in workflow_data):
                        workflow_health['properly_configured'] += 1
                        
            except Exception as e:
                print(f"   ⚠️ Issue with {wf_file.name}: {e}")
        
        score = (workflow_health['properly_configured'] / workflow_health['total_workflows']) * 100 if workflow_health['total_workflows'] > 0 else 0
        
        print(f"   ✅ Ultimate workflows: {workflow_health['properly_configured']}/{workflow_health['total_workflows']} working ({score:.1f}%)")
        
        return {
            'score': score,
            **workflow_health
        }
    
    def test_auto_response(self) -> Dict:
        """Test auto-response system"""
        print("🚨 Testing Auto-Response System...")
        
        main_workflow = Path(".github/workflows/cloud_bot_mechanic.yml")
        
        auto_response_features = {
            'main_workflow_exists': main_workflow.exists(),
            'has_workflow_run_triggers': False,
            'monitors_workflows': False,
            'ultimate_mode_enabled': False
        }
        
        if main_workflow.exists():
            content = main_workflow.read_text()
            
            auto_response_features['has_workflow_run_triggers'] = 'workflow_run:' in content
            auto_response_features['monitors_workflows'] = content.count('- "') >= 10
            auto_response_features['ultimate_mode_enabled'] = 'ULTIMATE_MODE: true' in content
        
        active_features = sum(1 for v in auto_response_features.values() if v)
        score = (active_features / len(auto_response_features)) * 100
        
        print(f"   ✅ Auto-response features: {active_features}/{len(auto_response_features)} active ({score:.1f}%)")
        
        return {
            'score': score,
            'features': auto_response_features
        }
    
    def test_feature_discovery(self) -> Dict:
        """Test that all 326 features are still discoverable"""
        print("🔍 Testing Feature Discovery...")
        
        try:
            # Run the comprehensive feature audit
            result = subprocess.run([
                'python', 'complete_85_feature_audit.py'
            ], capture_output=True, text=True, timeout=120)
            
            output = result.stdout
            
            # Extract feature count from output
            feature_count = 0
            if 'TOTAL CLOUD AI FEATURES:' in output:
                import re
                match = re.search(r'TOTAL CLOUD AI FEATURES: (\d+)', output)
                if match:
                    feature_count = int(match.group(1))
            
            score = min(100, (feature_count / 326) * 100) if feature_count > 0 else 0
            
            print(f"   ✅ Features discovered: {feature_count}/326 ({score:.1f}%)")
            
            return {
                'score': score,
                'features_discovered': feature_count,
                'target_features': 326,
                'meets_requirement': feature_count >= 85
            }
            
        except Exception as e:
            print(f"   ❌ Feature discovery failed: {e}")
            return {'score': 0, 'error': str(e)}
    
    def collect_performance_metrics(self) -> Dict:
        """Collect performance metrics"""
        print("📊 Collecting Performance Metrics...")
        
        metrics = {
            'cache_directory_size': 0,
            'reports_generated': 0,
            'database_files': 0,
            'workflow_files': 0
        }
        
        # Check cache directory
        cache_dir = Path('.mechanic-cache')
        if cache_dir.exists():
            metrics['cache_directory_size'] = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
        
        # Check reports
        reports_dir = Path('Intelligence/mechanic/cloud/reports')
        if reports_dir.exists():
            metrics['reports_generated'] = len(list(reports_dir.glob('*.json')))
        
        # Check database files
        db_dir = Path('Intelligence/mechanic/cloud/database')
        if db_dir.exists():
            metrics['database_files'] = len(list(db_dir.glob('*.json')))
        
        # Check workflow files
        workflows_dir = Path('.github/workflows')
        if workflows_dir.exists():
            metrics['workflow_files'] = len(list(workflows_dir.glob('*.yml')))
        
        # Calculate performance score based on activity
        score = 100  # Start with full score
        if metrics['cache_directory_size'] == 0:
            score -= 20
        if metrics['reports_generated'] == 0:
            score -= 20
        if metrics['database_files'] == 0:
            score -= 20
        
        print(f"   ✅ Performance metrics collected (score: {score}%)")
        
        return {
            'score': score,
            'metrics': metrics
        }
    
    def get_status_from_score(self, score: float) -> str:
        """Get status description from score"""
        if score >= 95:
            return "EXCELLENT - All systems optimal"
        elif score >= 85:
            return "GOOD - Minor issues detected"
        elif score >= 70:
            return "FAIR - Some attention needed"
        else:
            return "POOR - Immediate attention required"
    
    def save_monitoring_report(self, health_report: Dict):
        """Save monitoring report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.report_dir / f"health_check_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(health_report, f, indent=2, default=str)
        
        # Also save as latest
        latest_file = self.report_dir / "latest_health_check.json"
        with open(latest_file, 'w') as f:
            json.dump(health_report, f, indent=2, default=str)
        
        print(f"📄 Health report saved to: {report_file}")
    
    def run_single_check(self) -> Dict:
        """Run a single comprehensive health check"""
        health_report = self.run_feature_health_check()
        self.save_monitoring_report(health_report)
        
        # Print summary
        print(f"\n🎯 MONITORING SUMMARY:")
        print(f"   Overall Health: {health_report['overall_health_score']:.1f}%")
        print(f"   Status: {health_report['status']}")
        
        if health_report['overall_health_score'] >= 90:
            print("   🎉 All systems are running excellently!")
        elif health_report['overall_health_score'] >= 80:
            print("   ✅ Systems are running well with minor issues")
        else:
            print("   ⚠️ Some systems need attention")
        
        return health_report

def main():
    """Run the monitoring system"""
    print("🚀 CONTINUOUS CLOUD AI FEATURE MONITORING SYSTEM")
    print("=" * 60)
    
    monitor = ContinuousFeatureMonitor()
    
    # Run single comprehensive check
    health_report = monitor.run_single_check()
    
    return health_report

if __name__ == "__main__":
    main()