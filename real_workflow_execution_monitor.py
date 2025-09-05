#!/usr/bin/env python3
"""
REAL WORKFLOW EXECUTION MONITOR
Checks actual GitHub Actions execution history to see which workflows are really running
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import requests

class RealWorkflowExecutionMonitor:
    def __init__(self):
        self.github_token = os.environ.get('GITHUB_TOKEN')
        self.repo_owner = os.environ.get('GITHUB_REPOSITORY_OWNER', 'c-trading-bo')
        self.repo_name = os.environ.get('GITHUB_REPOSITORY', 'trading-bot-c-')
        self.workflows_dir = Path('.github/workflows')
        
    def get_github_api_headers(self):
        """Get headers for GitHub API requests"""
        if not self.github_token:
            # Try to get token from git config or env
            try:
                result = subprocess.run(['git', 'config', 'user.token'], capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    self.github_token = result.stdout.strip()
            except:
                pass
        
        if self.github_token:
            return {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
        else:
            return {'Accept': 'application/vnd.github.v3+json'}
    
    def get_workflow_runs_from_api(self) -> Dict:
        """Get all workflow runs from GitHub API"""
        headers = self.get_github_api_headers()
        
        try:
            # Get all workflows first
            workflows_url = f'https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/actions/workflows'
            response = requests.get(workflows_url, headers=headers)
            
            if response.status_code == 403:
                print("⚠️ API rate limited or access denied. Using local analysis.")
                return {'status': 'rate_limited'}
            elif response.status_code != 200:
                print(f"⚠️ API error: {response.status_code}. Using local analysis.")
                return {'status': 'api_error', 'code': response.status_code}
            
            workflows = response.json().get('workflows', [])
            
            # Get runs for each workflow
            all_runs = {}
            for workflow in workflows:
                workflow_name = Path(workflow['path']).name
                
                runs_url = f'https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/actions/workflows/{workflow["id"]}/runs'
                runs_response = requests.get(runs_url, headers=headers, params={'per_page': 20})
                
                if runs_response.status_code == 200:
                    runs_data = runs_response.json()
                    all_runs[workflow_name] = {
                        'workflow_id': workflow['id'],
                        'workflow_name': workflow['name'],
                        'state': workflow['state'],
                        'runs': runs_data.get('workflow_runs', [])
                    }
                else:
                    all_runs[workflow_name] = {
                        'workflow_id': workflow['id'],
                        'workflow_name': workflow['name'],
                        'state': workflow['state'],
                        'runs': [],
                        'error': f'Failed to get runs: {runs_response.status_code}'
                    }
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
            
            return {'status': 'success', 'data': all_runs}
            
        except Exception as e:
            print(f"⚠️ Error accessing GitHub API: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def analyze_workflow_activity(self, runs_data: Dict) -> Dict:
        """Analyze workflow activity from runs data"""
        analysis = {
            'total_workflows': 0,
            'active_workflows': [],
            'inactive_workflows': [],
            'problematic_workflows': [],
            'workflow_details': {}
        }
        
        cutoff_date = datetime.now() - timedelta(days=7)  # Last 7 days
        
        for workflow_name, workflow_data in runs_data.items():
            analysis['total_workflows'] += 1
            
            runs = workflow_data.get('runs', [])
            
            # Analyze this workflow
            workflow_analysis = {
                'total_runs': len(runs),
                'recent_runs': 0,
                'successful_runs': 0,
                'failed_runs': 0,
                'last_run_date': None,
                'last_successful_date': None,
                'success_rate': 0,
                'status': 'inactive'
            }
            
            for run in runs:
                run_date = datetime.fromisoformat(run['created_at'].replace('Z', '+00:00'))
                
                if run_date > cutoff_date:
                    workflow_analysis['recent_runs'] += 1
                
                if run.get('conclusion') == 'success':
                    workflow_analysis['successful_runs'] += 1
                    if not workflow_analysis['last_successful_date']:
                        workflow_analysis['last_successful_date'] = run['created_at']
                elif run.get('conclusion') in ['failure', 'cancelled', 'timed_out']:
                    workflow_analysis['failed_runs'] += 1
                
                if not workflow_analysis['last_run_date']:
                    workflow_analysis['last_run_date'] = run['created_at']
            
            # Calculate success rate
            if workflow_analysis['total_runs'] > 0:
                workflow_analysis['success_rate'] = workflow_analysis['successful_runs'] / workflow_analysis['total_runs']
            
            # Determine status
            if workflow_analysis['recent_runs'] > 0:
                if workflow_analysis['success_rate'] >= 0.8:
                    workflow_analysis['status'] = 'active_healthy'
                    analysis['active_workflows'].append(workflow_name)
                else:
                    workflow_analysis['status'] = 'active_problematic'
                    analysis['problematic_workflows'].append(workflow_name)
            else:
                workflow_analysis['status'] = 'inactive'
                analysis['inactive_workflows'].append(workflow_name)
            
            analysis['workflow_details'][workflow_name] = workflow_analysis
        
        return analysis
    
    def check_local_git_activity(self) -> Dict:
        """Check local git activity to infer workflow execution"""
        try:
            # Get recent commits that might have been from workflows
            result = subprocess.run([
                'git', 'log', '--oneline', '--since=7 days ago', '--grep=workflow', '--grep=action', '--grep=bot'
            ], capture_output=True, text=True, cwd=self.workflows_dir.parent)
            
            recent_commits = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            # Get workflow files modification dates
            workflow_files = list(self.workflows_dir.glob('*.yml'))
            file_activity = {}
            
            for wf_file in workflow_files:
                try:
                    # Check when file was last modified
                    stat = wf_file.stat()
                    mod_time = datetime.fromtimestamp(stat.st_mtime)
                    
                    # Check if file has been committed recently
                    result = subprocess.run([
                        'git', 'log', '-1', '--format=%ai', str(wf_file.relative_to(wf_file.parent.parent))
                    ], capture_output=True, text=True)
                    
                    last_commit = None
                    if result.returncode == 0 and result.stdout.strip():
                        last_commit = result.stdout.strip()
                    
                    file_activity[wf_file.name] = {
                        'last_modified': mod_time.isoformat(),
                        'last_commit': last_commit,
                        'recently_active': mod_time > datetime.now() - timedelta(days=7)
                    }
                    
                except Exception as e:
                    file_activity[wf_file.name] = {'error': str(e)}
            
            return {
                'recent_commits': recent_commits,
                'file_activity': file_activity
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def manually_trigger_test_workflow(self, workflow_name: str) -> Dict:
        """Manually trigger a workflow to test if it works"""
        headers = self.get_github_api_headers()
        
        if not self.github_token:
            return {'status': 'no_token', 'message': 'Cannot trigger workflow without GitHub token'}
        
        try:
            # First get the workflow ID
            workflows_url = f'https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/actions/workflows'
            response = requests.get(workflows_url, headers=headers)
            
            if response.status_code != 200:
                return {'status': 'error', 'message': f'Failed to get workflows: {response.status_code}'}
            
            workflows = response.json().get('workflows', [])
            workflow_id = None
            
            for workflow in workflows:
                if Path(workflow['path']).name == workflow_name:
                    workflow_id = workflow['id']
                    break
            
            if not workflow_id:
                return {'status': 'not_found', 'message': f'Workflow {workflow_name} not found'}
            
            # Trigger the workflow
            dispatch_url = f'https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/actions/workflows/{workflow_id}/dispatches'
            dispatch_data = {'ref': 'main'}  # or current branch
            
            response = requests.post(dispatch_url, headers=headers, json=dispatch_data)
            
            if response.status_code == 204:
                return {'status': 'triggered', 'message': f'Successfully triggered {workflow_name}'}
            else:
                return {'status': 'failed', 'message': f'Failed to trigger: {response.status_code}'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def run_comprehensive_execution_check(self) -> Dict:
        """Run comprehensive check of actual workflow execution"""
        print("🔍 REAL WORKFLOW EXECUTION ANALYSIS")
        print("=" * 60)
        
        # Get workflow files
        workflow_files = list(self.workflows_dir.glob('*.yml'))
        
        print(f"📋 Analyzing {len(workflow_files)} workflows for actual execution...")
        print()
        
        # Get API data
        print("🌐 Checking GitHub API for execution history...")
        api_data = self.get_workflow_runs_from_api()
        
        if api_data['status'] == 'success':
            activity_analysis = self.analyze_workflow_activity(api_data['data'])
            print("✅ Successfully retrieved execution data from GitHub API")
        else:
            print(f"⚠️ API check failed: {api_data.get('error', 'Unknown error')}")
            activity_analysis = None
        
        # Check local activity
        print("💾 Checking local git activity...")
        local_activity = self.check_local_git_activity()
        
        # Create comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'api_status': api_data['status'],
            'local_analysis': local_activity,
            'execution_summary': {}
        }
        
        if activity_analysis:
            report['execution_summary'] = activity_analysis
        
        # Print results
        print("\n📊 EXECUTION ANALYSIS RESULTS")
        print("=" * 60)
        
        if activity_analysis:
            active = activity_analysis['active_workflows']
            problematic = activity_analysis['problematic_workflows']
            inactive = activity_analysis['inactive_workflows']
            
            print(f"✅ ACTIVE WORKFLOWS ({len(active)}):")
            for wf in active:
                details = activity_analysis['workflow_details'][wf]
                last_run = details['last_run_date']
                success_rate = details['success_rate'] * 100
                print(f"   • {wf} - Last run: {last_run}, Success rate: {success_rate:.1f}%")
            
            print(f"\n⚠️ PROBLEMATIC WORKFLOWS ({len(problematic)}):")
            for wf in problematic:
                details = activity_analysis['workflow_details'][wf]
                last_run = details['last_run_date']
                success_rate = details['success_rate'] * 100
                print(f"   • {wf} - Last run: {last_run}, Success rate: {success_rate:.1f}%")
            
            print(f"\n❌ INACTIVE WORKFLOWS ({len(inactive)}):")
            for wf in inactive:
                details = activity_analysis['workflow_details'][wf]
                total_runs = details['total_runs']
                last_run = details['last_run_date'] or 'Never'
                print(f"   • {wf} - Total runs: {total_runs}, Last run: {last_run}")
            
            print(f"\n🎯 EXECUTION SUMMARY:")
            print(f"   Total workflows: {activity_analysis['total_workflows']}")
            print(f"   Currently active: {len(active)} ({len(active)/activity_analysis['total_workflows']*100:.1f}%)")
            print(f"   Problematic: {len(problematic)} ({len(problematic)/activity_analysis['total_workflows']*100:.1f}%)")
            print(f"   Inactive: {len(inactive)} ({len(inactive)/activity_analysis['total_workflows']*100:.1f}%)")
        
        else:
            print("⚠️ Could not retrieve execution data from GitHub API")
            print("Using local analysis only...")
            
            active_files = []
            inactive_files = []
            
            for wf_name, activity in local_activity.get('file_activity', {}).items():
                if activity.get('recently_active'):
                    active_files.append(wf_name)
                else:
                    inactive_files.append(wf_name)
            
            print(f"📁 Recently modified workflows: {len(active_files)}")
            for wf in active_files:
                print(f"   • {wf}")
            
            print(f"\n📁 Not recently modified: {len(inactive_files)}")
            for wf in inactive_files:
                print(f"   • {wf}")
        
        return report
    
    def save_execution_report(self, report: Dict):
        """Save execution report"""
        report_file = Path('real_workflow_execution_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Execution report saved to: {report_file}")

def main():
    """Run the real workflow execution monitor"""
    monitor = RealWorkflowExecutionMonitor()
    report = monitor.run_comprehensive_execution_check()
    monitor.save_execution_report(report)
    
    return report

if __name__ == "__main__":
    main()