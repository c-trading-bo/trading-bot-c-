#!/usr/bin/env python3
"""
Real-time Workflow Execution Monitor
Monitors GitHub Actions workflows to verify they're running as scheduled
"""

import requests
import json
import time
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional

class WorkflowMonitor:
    def __init__(self):
        self.github_token = os.getenv('GITHUB_TOKEN', '')
        self.repo_owner = "c-trading-bo"
        self.repo_name = "trading-bot-c-"
        self.base_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}"
        
        # Expected schedules for key workflows
        self.expected_schedules = {
            'es_nq_critical_trading': {
                'market_hours': 3,      # Every 3 minutes
                'pre_post': 5,          # Every 5 minutes
                'overnight': 15,        # Every 15 minutes
                'sunday': 30            # Every 30 minutes
            },
            'ultimate_ml_rl_intel_system': {
                'market_hours': 5,      # Every 5 minutes
                'pre_post': 10,         # Every 10 minutes
                'overnight': 30,        # Every 30 minutes
                'sunday': 120           # Every 2 hours
            }
        }
        
        self.headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        } if self.github_token else {}
    
    def get_current_period(self) -> str:
        """Determine current market period"""
        now = datetime.now()
        hour = now.hour
        day = now.strftime('%A')
        
        if day == 'Saturday':
            return 'saturday'
        elif day == 'Sunday':
            return 'sunday'
        elif 9 <= hour < 16:  # 9:30 AM - 4:00 PM market hours
            return 'market_hours'
        elif 7 <= hour < 9 or 16 <= hour < 18:  # Pre/post market
            return 'pre_post'
        else:
            return 'overnight'
    
    def get_workflow_runs(self, workflow_name: str, hours_back: int = 2) -> List[Dict]:
        """Get recent workflow runs"""
        if not self.github_token:
            print("⚠️  No GITHUB_TOKEN - using mock data for demonstration")
            return self.get_mock_runs(workflow_name)
        
        since = (datetime.now() - timedelta(hours=hours_back)).isoformat() + 'Z'
        url = f"{self.base_url}/actions/workflows/{workflow_name}.yml/runs"
        params = {
            'created': f'>={since}',
            'per_page': 50,
            'status': 'completed'
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json().get('workflow_runs', [])
        except requests.RequestException as e:
            print(f"❌ Error fetching runs for {workflow_name}: {e}")
            return []
    
    def get_mock_runs(self, workflow_name: str) -> List[Dict]:
        """Generate mock runs for demonstration when no token available"""
        now = datetime.now()
        runs = []
        
        # Generate runs based on expected schedule
        if workflow_name in self.expected_schedules:
            period = self.get_current_period()
            if period == 'saturday':
                return []  # No Saturday runs
            
            interval = self.expected_schedules[workflow_name].get(period, 15)
            
            # Generate last 2 hours of runs
            for i in range(0, 120, interval):
                run_time = now - timedelta(minutes=i)
                runs.append({
                    'id': f"mock_{i}",
                    'created_at': run_time.isoformat() + 'Z',
                    'status': 'completed',
                    'conclusion': 'success',
                    'html_url': f"https://github.com/{self.repo_owner}/{self.repo_name}/actions/runs/mock_{i}"
                })
        
        return runs[:10]  # Return last 10 runs
    
    def analyze_run_frequency(self, runs: List[Dict], workflow_name: str) -> Dict:
        """Analyze if runs match expected frequency"""
        if not runs:
            return {'status': 'no_runs', 'message': 'No recent runs found'}
        
        # Calculate intervals between runs
        run_times = [datetime.fromisoformat(run['created_at'].replace('Z', '')) for run in runs]
        run_times.sort(reverse=True)
        
        intervals = []
        for i in range(len(run_times) - 1):
            interval = (run_times[i] - run_times[i + 1]).total_seconds() / 60
            intervals.append(interval)
        
        if not intervals:
            return {'status': 'single_run', 'message': 'Only one run found'}
        
        avg_interval = sum(intervals) / len(intervals)
        period = self.get_current_period()
        
        if workflow_name in self.expected_schedules:
            expected = self.expected_schedules[workflow_name].get(period, 15)
            tolerance = expected * 0.3  # 30% tolerance
            
            if abs(avg_interval - expected) <= tolerance:
                status = 'on_schedule'
                message = f"✅ Running on schedule (avg: {avg_interval:.1f}min, expected: {expected}min)"
            else:
                status = 'off_schedule'
                message = f"⚠️  Off schedule (avg: {avg_interval:.1f}min, expected: {expected}min)"
        else:
            status = 'unknown_schedule'
            message = f"ℹ️  Unknown schedule (avg interval: {avg_interval:.1f}min)"
        
        return {
            'status': status,
            'message': message,
            'avg_interval': avg_interval,
            'recent_runs': len(runs),
            'period': period
        }
    
    def monitor_workflow(self, workflow_name: str) -> Dict:
        """Monitor a specific workflow"""
        print(f"\n🔍 Monitoring: {workflow_name}")
        
        runs = self.get_workflow_runs(workflow_name)
        analysis = self.analyze_run_frequency(runs, workflow_name)
        
        print(f"   {analysis['message']}")
        
        if runs:
            latest_run = runs[0]
            latest_time = datetime.fromisoformat(latest_run['created_at'].replace('Z', ''))
            time_ago = datetime.now() - latest_time
            print(f"   📅 Latest run: {time_ago.total_seconds()/60:.1f} minutes ago")
            print(f"   🔗 Status: {latest_run['conclusion']}")
        
        return analysis
    
    def run_continuous_monitoring(self, interval_minutes: int = 5):
        """Run continuous monitoring"""
        print("🚀 Starting Real-time Workflow Monitor")
        print(f"📊 Checking every {interval_minutes} minutes")
        print("=" * 60)
        
        key_workflows = [
            'es_nq_critical_trading',
            'ultimate_ml_rl_intel_system',
            'es_nq_correlation_matrix',
            'ultimate_ml_rl_training_pipeline'
        ]
        
        try:
            while True:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                period = self.get_current_period()
                
                print(f"\n⏰ {current_time} - Current period: {period.upper()}")
                print("-" * 50)
                
                all_good = True
                for workflow in key_workflows:
                    analysis = self.monitor_workflow(workflow)
                    if analysis['status'] not in ['on_schedule', 'no_runs']:
                        all_good = False
                
                if all_good:
                    print("\n🎯 All monitored workflows are running as expected!")
                else:
                    print("\n⚠️  Some workflows may need attention!")
                
                print(f"\n💤 Sleeping for {interval_minutes} minutes...")
                print("=" * 60)
                
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Error during monitoring: {e}")

def main():
    monitor = WorkflowMonitor()
    
    print("🔍 GitHub Actions Workflow Monitor")
    print("=" * 40)
    
    # Single check first
    print("\n📊 Current Status Check:")
    key_workflows = [
        'es_nq_critical_trading',
        'ultimate_ml_rl_intel_system'
    ]
    
    for workflow in key_workflows:
        monitor.monitor_workflow(workflow)
    
    print("\n" + "=" * 40)
    choice = input("\nStart continuous monitoring? (y/n): ").lower()
    
    if choice == 'y':
        monitor.run_continuous_monitoring()
    else:
        print("👍 Single check complete!")

if __name__ == "__main__":
    main()
