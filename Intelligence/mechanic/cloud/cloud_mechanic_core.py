#!/usr/bin/env python3
"""
CLOUD BOT MECHANIC CORE - Ultimate System
Monitors, analyzes, repairs, and optimizes everything in the cloud
"""

import os
import sys
import json
import yaml
import subprocess
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re
import hashlib
import traceback
from croniter import croniter
import pytz

class CloudBotMechanic:
    def __init__(self):
        self.version = "3.0-CLOUD-ULTIMATE"
        self.start_time = datetime.utcnow()
        self.github_token = os.environ.get('GITHUB_TOKEN', '')
        self.repo_owner = os.environ.get('GITHUB_REPOSITORY_OWNER', '')
        self.repo_name = os.environ.get('GITHUB_REPOSITORY', '').split('/')[-1]
        self.operation_mode = os.environ.get('INPUT_MODE', 'auto')

        # Paths
        self.base_path = Path.cwd()
        self.workflow_path = Path('.github/workflows')
        self.cloud_path = Path('Intelligence/mechanic/cloud')
        self.db_path = self.cloud_path / 'database'

        # Create directories
        self.db_path.mkdir(parents=True, exist_ok=True)
        (self.cloud_path / 'reports').mkdir(exist_ok=True)
        (self.cloud_path / 'alerts').mkdir(exist_ok=True)
        (self.cloud_path / 'logs').mkdir(exist_ok=True)

        # Databases
        self.workflow_db = self.db_path / 'workflows.json'
        self.performance_db = self.db_path / 'performance.json'
        self.issues_db = self.db_path / 'issues.json'
        self.repairs_db = self.db_path / 'repairs.json'

        # Load databases
        self.workflows = self.load_json(self.workflow_db, {})
        self.performance = self.load_json(self.performance_db, {})
        self.issues = self.load_json(self.issues_db, [])
        self.repairs = self.load_json(self.repairs_db, [])

        # GitHub API headers
        self.headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }

        print(f"🌩️ Cloud Bot Mechanic v{self.version} initialized")
        print(f"📍 Repository: {self.repo_owner}/{self.repo_name}")
        print(f"🔧 Mode: {self.operation_mode}")

    def load_json(self, path: Path, default: Any) -> Any:
        """Load JSON with default fallback"""
        try:
            if path.exists():
                with open(path, 'r') as f:
                    return json.load(f)
        except:
            pass
        return default

    def save_json(self, path: Path, data: Any):
        """Save JSON data"""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Failed to save {path}: {e}")

    def analyze_all_workflows(self) -> Dict:
        """Deep analysis of all workflows"""
        print("\n🔍 ANALYZING ALL WORKFLOWS...")

        analysis = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_workflows': 0,
            'healthy_workflows': 0,
            'broken_workflows': [],
            'slow_workflows': [],
            'expensive_workflows': [],
            'schedule_conflicts': [],
            'optimization_opportunities': [],
            'total_monthly_minutes': 0,
            'issues_found': []
        }

        # Get all workflow files
        workflow_files = []
        if self.workflow_path.exists():
            workflow_files = list(self.workflow_path.glob('*.yml')) + list(self.workflow_path.glob('*.yaml'))
        
        analysis['total_workflows'] = len(workflow_files)

        for wf_file in workflow_files:
            self._analyze_workflow(wf_file, analysis)

        # Calculate minute usage
        self._calculate_minute_usage(analysis)

        # Save analysis
        self.workflows['last_analysis'] = analysis
        self.save_json(self.workflow_db, self.workflows)

        self._print_analysis_report(analysis)
        return analysis

    def _analyze_workflow(self, wf_file: Path, analysis: Dict):
        """Analyze individual workflow file"""
        try:
            # Try multiple encodings to handle special characters
            workflow = None
            raw_content = None
            for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
                try:
                    with open(wf_file, 'r', encoding=encoding) as f:
                        raw_content = f.read()
                        workflow = yaml.safe_load(raw_content)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue

            wf_name = wf_file.name
            wf_data = {
                'name': workflow.get('name', wf_name) if workflow else wf_name,
                'file': wf_name,
                'issues': [],
                'optimizations': [],
                'schedules': [],
                'estimated_minutes': 5,
                'runs_per_month': 0,
                'jobs': {}
            }

            # Check for YAML syntax issues
            if raw_content and 'true:' in raw_content and 'on:' not in raw_content:
                wf_data['issues'].append('YAML syntax error: "true:" should be "on:"')
                analysis['broken_workflows'].append(wf_name)
            elif workflow and True in workflow:  # Parsed YAML has True key instead of 'on'
                wf_data['issues'].append('Invalid workflow trigger: missing "on:" section')
                analysis['broken_workflows'].append(wf_name)

            # Check workflow structure
            elif not workflow:
                wf_data['issues'].append('Empty workflow file or encoding issue')
                analysis['broken_workflows'].append(wf_name)
            elif 'jobs' not in workflow:
                wf_data['issues'].append('No jobs defined')
                analysis['broken_workflows'].append(wf_name)
            else:
                # Analyze triggers
                self._analyze_triggers(workflow.get('on', {}), wf_data)

                # Analyze jobs
                for job_name, job_config in workflow.get('jobs', {}).items():
                    self._analyze_job(job_name, job_config, wf_data)

                if not wf_data['issues']:
                    analysis['healthy_workflows'] += 1

            # Store workflow data
            self.workflows[wf_name] = wf_data

        except Exception as e:
            analysis['broken_workflows'].append(wf_file.name)
            analysis['issues_found'].append(f"{wf_file.name}: Analysis error - {e}")

    def _analyze_triggers(self, triggers: Dict, wf_data: Dict):
        """Analyze workflow triggers"""
        if isinstance(triggers, dict):
            # Check schedules
            if 'schedule' in triggers:
                for schedule in triggers['schedule']:
                    if 'cron' in schedule:
                        cron_expr = schedule['cron']
                        wf_data['schedules'].append(cron_expr)
                        try:
                            croniter(cron_expr)
                            wf_data['runs_per_month'] = self._calculate_cron_runs(cron_expr)
                        except:
                            wf_data['issues'].append(f"Invalid cron: {cron_expr}")

    def _analyze_job(self, job_name: str, job_config: Dict, wf_data: Dict):
        """Analyze individual job"""
        job_data = {
            'timeout': job_config.get('timeout-minutes', 360),
            'runs_on': job_config.get('runs-on', 'ubuntu-latest'),
            'steps': len(job_config.get('steps', [])),
            'estimated_minutes': 5
        }

        # Estimate minutes
        job_data['estimated_minutes'] = min(job_data['timeout'], job_data['steps'] * 2)
        wf_data['estimated_minutes'] += job_data['estimated_minutes']
        wf_data['jobs'][job_name] = job_data

    def _calculate_cron_runs(self, cron_expr: str) -> int:
        """Calculate number of runs per month for cron expression"""
        try:
            now = datetime.utcnow()
            cron = croniter(cron_expr, now)
            runs = 0
            end_date = now + timedelta(days=30)

            while True:
                next_run = cron.get_next(datetime)
                if next_run > end_date:
                    break
                runs += 1
                if runs > 10000:  # Safety limit
                    break

            return runs
        except:
            return 0

    def _calculate_minute_usage(self, analysis: Dict):
        """Calculate total minute usage"""
        total_minutes = 0

        for wf_name, wf_data in self.workflows.items():
            if isinstance(wf_data, dict):
                runs_per_month = wf_data.get('runs_per_month', 0)
                estimated_minutes = wf_data.get('estimated_minutes', 5)
                workflow_monthly_minutes = runs_per_month * estimated_minutes
                total_minutes += workflow_monthly_minutes

        analysis['total_monthly_minutes'] = total_minutes

        if total_minutes > 20000:
            analysis['issues_found'].append(f"CRITICAL: Estimated {total_minutes} minutes exceeds 20,000 limit!")

    def _print_analysis_report(self, analysis: Dict):
        """Print analysis report"""
        print(f"\n📊 ANALYSIS COMPLETE")
        print(f"   Total workflows: {analysis['total_workflows']}")
        print(f"   Healthy workflows: {analysis['healthy_workflows']}")
        print(f"   Broken workflows: {len(analysis['broken_workflows'])}")
        print(f"   Monthly minutes: {analysis['total_monthly_minutes']}")
        
        if analysis['issues_found']:
            print(f"   Issues found: {len(analysis['issues_found'])}")

    def run(self):
        """Main run function"""
        print("🌩️ CLOUD BOT MECHANIC STARTING...")
        
        # Analyze workflows
        analysis = self.analyze_all_workflows()
        
        # Generate report
        report = self.generate_report(analysis)
        
        # Generate alerts if needed
        self.generate_alerts(analysis)
        
        # Set outputs for GitHub Actions
        self.set_github_outputs(analysis)
        
        print(f"\n✅ Cloud Bot Mechanic completed in {(datetime.utcnow() - self.start_time).total_seconds():.1f}s")

    def generate_report(self, analysis: Dict) -> Dict:
        """Generate comprehensive report"""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'version': self.version,
            'repository': f"{self.repo_owner}/{self.repo_name}",
            'analysis': analysis
        }

        # Save report
        report_path = self.cloud_path / 'reports' / f'report_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.json'
        self.save_json(report_path, report)
        self.save_json(self.cloud_path / 'reports' / 'latest_report.json', report)

        return report

    def generate_alerts(self, analysis: Dict):
        """Generate alerts for critical issues"""
        if analysis['total_monthly_minutes'] > 20000:
            alert = {
                'level': 'CRITICAL',
                'title': 'Minute Usage Exceeds Limit!',
                'message': f"Using {analysis['total_monthly_minutes']}/20000 minutes",
                'timestamp': datetime.utcnow().isoformat()
            }
            
            alert_path = self.cloud_path / 'alerts' / 'critical.json'
            self.save_json(alert_path, alert)

    def set_github_outputs(self, analysis: Dict):
        """Set GitHub Actions outputs"""
        try:
            # Check if we have issues that need repair
            has_issues = len(analysis['broken_workflows']) > 0 or len(analysis['issues_found']) > 0
            needs_optimization = analysis['total_monthly_minutes'] > 15000
            
            # Write to GitHub outputs
            if os.environ.get('GITHUB_OUTPUT'):
                with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                    f.write(f"repairs_needed={'true' if has_issues else 'false'}\n")
                    f.write(f"needs_optimization={'true' if needs_optimization else 'false'}\n")
                    f.write(f"changes_made=false\n")  # Will be updated by repair scripts
                    f.write(f"total_workflows={analysis['total_workflows']}\n")
                    f.write(f"monthly_minutes={analysis['total_monthly_minutes']}\n")
            
            print(f"📤 Set GitHub outputs: repairs_needed={has_issues}, needs_optimization={needs_optimization}")
            
        except Exception as e:
            print(f"⚠️ Failed to set GitHub outputs: {e}")

if __name__ == "__main__":
    try:
        mechanic = CloudBotMechanic()
        mechanic.run()
    except Exception as e:
        print(f"❌ Cloud Bot Mechanic failed: {e}")
        traceback.print_exc()
        sys.exit(1)
