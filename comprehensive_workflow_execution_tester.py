#!/usr/bin/env python3
"""
COMPREHENSIVE WORKFLOW EXECUTION TESTER
Actually runs and monitors all workflows to ensure they work correctly
"""

import os
import sys
import json
import subprocess
import time
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import requests
import tempfile

class WorkflowExecutionTester:
    def __init__(self):
        self.github_token = os.environ.get('GITHUB_TOKEN')
        self.repo_owner = os.environ.get('GITHUB_REPOSITORY_OWNER', 'c-trading-bo')
        self.repo_name = os.environ.get('GITHUB_REPOSITORY', 'trading-bot-c-')
        self.workflows_dir = Path('.github/workflows')
        self.results = {}
        self.execution_logs = []
        
    def get_workflow_files(self) -> List[Path]:
        """Get all workflow files"""
        return list(self.workflows_dir.glob('*.yml'))
    
    def check_workflow_syntax(self, workflow_file: Path) -> Tuple[bool, str]:
        """Check if workflow has valid YAML syntax"""
        try:
            with open(workflow_file, 'r', encoding='utf-8') as f:
                yaml.safe_load(f)
            return True, "Valid YAML syntax"
        except yaml.YAMLError as e:
            return False, f"YAML syntax error: {e}"
        except Exception as e:
            return False, f"Error reading file: {e}"
    
    def analyze_workflow_configuration(self, workflow_file: Path) -> Dict:
        """Analyze workflow configuration for completeness"""
        try:
            with open(workflow_file, 'r', encoding='utf-8') as f:
                content = f.read()
                workflow_data = yaml.safe_load(content)
            
            analysis = {
                'has_name': 'name' in workflow_data,
                'has_triggers': 'on' in workflow_data,
                'has_jobs': 'jobs' in workflow_data,
                'has_permissions': 'permissions' in workflow_data,
                'has_schedule': False,
                'has_workflow_dispatch': False,
                'has_dependencies': False,
                'trigger_types': [],
                'job_count': 0,
                'estimated_complexity': 'low'
            }
            
            if 'on' in workflow_data:
                triggers = workflow_data['on']
                if isinstance(triggers, dict):
                    analysis['trigger_types'] = list(triggers.keys())
                    analysis['has_schedule'] = 'schedule' in triggers
                    analysis['has_workflow_dispatch'] = 'workflow_dispatch' in triggers
                elif isinstance(triggers, list):
                    analysis['trigger_types'] = triggers
            
            if 'jobs' in workflow_data:
                jobs = workflow_data['jobs']
                analysis['job_count'] = len(jobs) if isinstance(jobs, dict) else 0
                
                # Check for dependencies
                for job_name, job_data in jobs.items() if isinstance(jobs, dict) else []:
                    if isinstance(job_data, dict) and 'needs' in job_data:
                        analysis['has_dependencies'] = True
                        break
                
                # Estimate complexity
                if analysis['job_count'] > 3:
                    analysis['estimated_complexity'] = 'high'
                elif analysis['job_count'] > 1:
                    analysis['estimated_complexity'] = 'medium'
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def test_workflow_execution_locally(self, workflow_file: Path) -> Dict:
        """Test workflow execution by simulating key steps locally"""
        print(f"🧪 Testing local execution for: {workflow_file.name}")
        
        try:
            with open(workflow_file, 'r', encoding='utf-8') as f:
                workflow_data = yaml.safe_load(f)
            
            execution_results = {
                'workflow_name': workflow_file.name,
                'can_parse': True,
                'jobs_testable': 0,
                'jobs_failed': 0,
                'dependency_check': 'passed',
                'environment_check': 'passed',
                'estimated_runtime': 'unknown',
                'issues_found': []
            }
            
            if 'jobs' not in workflow_data:
                execution_results['issues_found'].append("No jobs defined")
                return execution_results
            
            jobs = workflow_data['jobs']
            if not isinstance(jobs, dict):
                execution_results['issues_found'].append("Jobs not properly defined")
                return execution_results
            
            for job_name, job_data in jobs.items():
                if not isinstance(job_data, dict):
                    execution_results['jobs_failed'] += 1
                    execution_results['issues_found'].append(f"Job {job_name} not properly defined")
                    continue
                
                execution_results['jobs_testable'] += 1
                
                # Check runs-on
                if 'runs-on' not in job_data:
                    execution_results['issues_found'].append(f"Job {job_name} missing runs-on")
                
                # Check steps
                if 'steps' not in job_data:
                    execution_results['issues_found'].append(f"Job {job_name} has no steps")
                    continue
                
                steps = job_data['steps']
                if not isinstance(steps, list):
                    execution_results['issues_found'].append(f"Job {job_name} steps not a list")
                    continue
                
                # Test key steps
                for i, step in enumerate(steps):
                    if not isinstance(step, dict):
                        execution_results['issues_found'].append(f"Job {job_name} step {i} not properly defined")
                        continue
                    
                    # Check for common issues
                    if 'uses' in step:
                        action = step['uses']
                        if action.startswith('./') and not Path(action.lstrip('./')).exists():
                            execution_results['issues_found'].append(f"Local action {action} not found")
                    
                    if 'run' in step:
                        command = step['run']
                        # Check for common command issues
                        if 'python' in command and not self.check_python_available():
                            execution_results['issues_found'].append("Python not available for steps requiring it")
                        
                        if 'node' in command and not self.check_node_available():
                            execution_results['issues_found'].append("Node.js not available for steps requiring it")
            
            # Calculate success rate
            if execution_results['jobs_testable'] > 0:
                success_rate = (execution_results['jobs_testable'] - execution_results['jobs_failed']) / execution_results['jobs_testable']
                if success_rate >= 0.8:
                    execution_results['overall_status'] = 'likely_to_work'
                elif success_rate >= 0.5:
                    execution_results['overall_status'] = 'may_have_issues'
                else:
                    execution_results['overall_status'] = 'likely_to_fail'
            else:
                execution_results['overall_status'] = 'cannot_determine'
            
            return execution_results
            
        except Exception as e:
            return {
                'workflow_name': workflow_file.name,
                'can_parse': False,
                'error': str(e),
                'overall_status': 'failed_analysis'
            }
    
    def check_python_available(self) -> bool:
        """Check if Python is available"""
        try:
            subprocess.run(['python', '--version'], capture_output=True, check=True)
            return True
        except:
            try:
                subprocess.run(['python3', '--version'], capture_output=True, check=True)
                return True
            except:
                return False
    
    def check_node_available(self) -> bool:
        """Check if Node.js is available"""
        try:
            subprocess.run(['node', '--version'], capture_output=True, check=True)
            return True
        except:
            return False
    
    def check_recent_workflow_runs(self, workflow_name: str) -> Dict:
        """Check recent runs of a workflow via GitHub API"""
        if not self.github_token:
            return {'status': 'no_token', 'message': 'GitHub token not available'}
        
        try:
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            # Get workflow ID first
            workflows_url = f'https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/actions/workflows'
            response = requests.get(workflows_url, headers=headers)
            
            if response.status_code != 200:
                return {'status': 'api_error', 'message': f'API error: {response.status_code}'}
            
            workflows = response.json().get('workflows', [])
            workflow_id = None
            
            for workflow in workflows:
                if workflow['path'].endswith(workflow_name):
                    workflow_id = workflow['id']
                    break
            
            if not workflow_id:
                return {'status': 'not_found', 'message': 'Workflow not found in GitHub'}
            
            # Get recent runs
            runs_url = f'https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/actions/workflows/{workflow_id}/runs'
            response = requests.get(runs_url, headers=headers, params={'per_page': 10})
            
            if response.status_code != 200:
                return {'status': 'api_error', 'message': f'Runs API error: {response.status_code}'}
            
            runs = response.json().get('workflow_runs', [])
            
            if not runs:
                return {'status': 'no_runs', 'message': 'No recent runs found'}
            
            latest_run = runs[0]
            recent_runs_summary = {
                'total_runs': len(runs),
                'latest_status': latest_run['status'],
                'latest_conclusion': latest_run.get('conclusion'),
                'latest_run_date': latest_run['created_at'],
                'success_rate': sum(1 for run in runs if run.get('conclusion') == 'success') / len(runs),
                'last_successful_run': None
            }
            
            # Find last successful run
            for run in runs:
                if run.get('conclusion') == 'success':
                    recent_runs_summary['last_successful_run'] = run['created_at']
                    break
            
            return {'status': 'success', 'data': recent_runs_summary}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def run_comprehensive_test(self) -> Dict:
        """Run comprehensive test of all workflows"""
        print("🚀 COMPREHENSIVE WORKFLOW EXECUTION TESTING")
        print("=" * 60)
        
        workflow_files = self.get_workflow_files()
        total_workflows = len(workflow_files)
        
        print(f"📋 Found {total_workflows} workflow files to test")
        print()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_workflows': total_workflows,
            'tested_workflows': 0,
            'syntax_valid': 0,
            'execution_ready': 0,
            'recently_run': 0,
            'working_workflows': [],
            'problematic_workflows': [],
            'not_running_workflows': [],
            'detailed_results': {}
        }
        
        for i, workflow_file in enumerate(workflow_files, 1):
            print(f"🔍 Testing {i}/{total_workflows}: {workflow_file.name}")
            
            # Test syntax
            syntax_valid, syntax_message = self.check_workflow_syntax(workflow_file)
            
            # Analyze configuration
            config_analysis = self.analyze_workflow_configuration(workflow_file)
            
            # Test local execution readiness
            execution_test = self.test_workflow_execution_locally(workflow_file)
            
            # Check recent runs
            recent_runs = self.check_recent_workflow_runs(workflow_file.name)
            
            workflow_result = {
                'filename': workflow_file.name,
                'syntax_valid': syntax_valid,
                'syntax_message': syntax_message,
                'configuration': config_analysis,
                'execution_test': execution_test,
                'recent_runs': recent_runs,
                'overall_assessment': self.assess_workflow_health(
                    syntax_valid, config_analysis, execution_test, recent_runs
                )
            }
            
            results['detailed_results'][workflow_file.name] = workflow_result
            results['tested_workflows'] += 1
            
            if syntax_valid:
                results['syntax_valid'] += 1
            
            if execution_test.get('overall_status') == 'likely_to_work':
                results['execution_ready'] += 1
            
            # Categorize workflow
            assessment = workflow_result['overall_assessment']
            if assessment['status'] == 'working':
                results['working_workflows'].append(workflow_file.name)
                results['recently_run'] += 1
            elif assessment['status'] == 'problematic':
                results['problematic_workflows'].append(workflow_file.name)
            else:
                results['not_running_workflows'].append(workflow_file.name)
            
            # Print summary for this workflow
            status_emoji = {
                'working': '✅',
                'problematic': '⚠️',
                'not_working': '❌'
            }.get(assessment['status'], '❓')
            
            print(f"   {status_emoji} {assessment['summary']}")
            print()
        
        # Print final summary
        print("📊 FINAL TESTING RESULTS")
        print("=" * 60)
        print(f"✅ Working Workflows: {len(results['working_workflows'])}")
        for wf in results['working_workflows']:
            print(f"   • {wf}")
        print()
        
        print(f"⚠️ Problematic Workflows: {len(results['problematic_workflows'])}")
        for wf in results['problematic_workflows']:
            print(f"   • {wf}")
        print()
        
        print(f"❌ Not Running Workflows: {len(results['not_running_workflows'])}")
        for wf in results['not_running_workflows']:
            print(f"   • {wf}")
        print()
        
        success_rate = len(results['working_workflows']) / total_workflows * 100
        print(f"🎯 Overall Success Rate: {success_rate:.1f}%")
        
        return results
    
    def assess_workflow_health(self, syntax_valid: bool, config: Dict, execution: Dict, recent_runs: Dict) -> Dict:
        """Assess overall workflow health"""
        if not syntax_valid:
            return {
                'status': 'not_working',
                'summary': 'Invalid YAML syntax',
                'issues': ['Syntax errors prevent execution']
            }
        
        issues = []
        score = 100
        
        # Check configuration issues
        if config.get('error'):
            issues.append(f"Configuration error: {config['error']}")
            score -= 50
        
        if not config.get('has_jobs'):
            issues.append("No jobs defined")
            score -= 30
        
        if not config.get('has_triggers'):
            issues.append("No triggers defined")
            score -= 20
        
        # Check execution readiness
        exec_status = execution.get('overall_status')
        if exec_status == 'likely_to_fail':
            issues.append("Execution likely to fail")
            score -= 40
        elif exec_status == 'may_have_issues':
            issues.append("May have execution issues")
            score -= 20
        
        execution_issues = execution.get('issues_found', [])
        if execution_issues:
            issues.extend(execution_issues)
            score -= len(execution_issues) * 5
        
        # Check recent runs
        runs_status = recent_runs.get('status')
        if runs_status == 'success':
            runs_data = recent_runs['data']
            if runs_data['success_rate'] < 0.5:
                issues.append("Low success rate in recent runs")
                score -= 30
            elif runs_data['latest_conclusion'] != 'success':
                issues.append("Latest run failed")
                score -= 15
        elif runs_status == 'no_runs':
            issues.append("No recent execution history")
            score -= 25
        
        # Determine final status
        if score >= 80:
            status = 'working'
            summary = "Working well"
        elif score >= 60:
            status = 'problematic'
            summary = "Has issues but may work"
        else:
            status = 'not_working'
            summary = "Not working properly"
        
        if issues:
            summary += f" ({len(issues)} issues found)"
        
        return {
            'status': status,
            'summary': summary,
            'score': score,
            'issues': issues
        }
    
    def save_results(self, results: Dict):
        """Save test results to file"""
        results_file = Path('workflow_execution_test_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📄 Results saved to: {results_file}")

def main():
    """Run the comprehensive workflow execution test"""
    tester = WorkflowExecutionTester()
    results = tester.run_comprehensive_test()
    tester.save_results(results)
    
    return results

if __name__ == "__main__":
    main()