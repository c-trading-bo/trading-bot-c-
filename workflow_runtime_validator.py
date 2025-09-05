#!/usr/bin/env python3
"""
WORKFLOW RUNTIME VALIDATION SYSTEM
Tests if workflows are actually executing their intended functions
"""

import os
import sys
import json
import subprocess
import time
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

class WorkflowRuntimeValidator:
    def __init__(self):
        self.workflows_dir = Path('.github/workflows')
        self.test_results = {}
        
    def extract_key_commands_from_workflow(self, workflow_file: Path) -> List[str]:
        """Extract key commands that the workflow runs"""
        try:
            with open(workflow_file, 'r', encoding='utf-8') as f:
                workflow_data = yaml.safe_load(f)
            
            commands = []
            
            if 'jobs' in workflow_data:
                for job_name, job_data in workflow_data['jobs'].items():
                    if isinstance(job_data, dict) and 'steps' in job_data:
                        for step in job_data['steps']:
                            if isinstance(step, dict):
                                if 'run' in step:
                                    commands.append(step['run'])
                                elif 'uses' in step and step['uses'].startswith('./'):
                                    # Local action
                                    action_path = step['uses'].lstrip('./')
                                    if Path(action_path).exists():
                                        commands.append(f"Local action: {action_path}")
            
            return commands
            
        except Exception as e:
            return [f"Error parsing workflow: {e}"]
    
    def test_python_scripts_in_workflow(self, commands: List[str]) -> Dict:
        """Test if Python scripts mentioned in workflow actually work"""
        results = {
            'python_scripts_found': 0,
            'python_scripts_working': 0,
            'script_test_results': {}
        }
        
        for command in commands:
            if 'python' in command.lower():
                results['python_scripts_found'] += 1
                
                # Extract Python script name
                import re
                python_matches = re.findall(r'python[3]?\s+([^\s]+\.py)', command)
                
                for script_path in python_matches:
                    if Path(script_path).exists():
                        # Test the script
                        try:
                            # Run with --help or dry run if possible
                            test_result = subprocess.run([
                                'python', script_path, '--help'
                            ], capture_output=True, text=True, timeout=30)
                            
                            if test_result.returncode != 0:
                                # Try without --help
                                test_result = subprocess.run([
                                    'python', '-c', f'import sys; sys.path.append("."); import importlib.util; spec = importlib.util.spec_from_file_location("test", "{script_path}"); module = importlib.util.module_from_spec(spec); print("Script is importable")'
                                ], capture_output=True, text=True, timeout=30)
                            
                            if test_result.returncode == 0:
                                results['python_scripts_working'] += 1
                                results['script_test_results'][script_path] = 'working'
                            else:
                                results['script_test_results'][script_path] = f'failed: {test_result.stderr}'
                                
                        except subprocess.TimeoutExpired:
                            results['script_test_results'][script_path] = 'timeout'
                        except Exception as e:
                            results['script_test_results'][script_path] = f'error: {e}'
                    else:
                        results['script_test_results'][script_path] = 'file_not_found'
        
        return results
    
    def check_workflow_dependencies(self, workflow_file: Path) -> Dict:
        """Check if workflow dependencies are available"""
        try:
            with open(workflow_file, 'r', encoding='utf-8') as f:
                content = f.read()
                workflow_data = yaml.safe_load(content)
            
            dependency_check = {
                'uses_python': 'python' in content.lower(),
                'uses_node': 'node' in content.lower() or 'npm' in content.lower(),
                'uses_docker': 'docker' in content.lower(),
                'uses_custom_actions': False,
                'missing_dependencies': [],
                'dependency_score': 100
            }
            
            # Check for custom actions
            if 'jobs' in workflow_data:
                for job_name, job_data in workflow_data['jobs'].items():
                    if isinstance(job_data, dict) and 'steps' in job_data:
                        for step in job_data['steps']:
                            if isinstance(step, dict) and 'uses' in step:
                                action = step['uses']
                                if action.startswith('./'):
                                    dependency_check['uses_custom_actions'] = True
                                    action_path = Path(action.lstrip('./'))
                                    if not action_path.exists():
                                        dependency_check['missing_dependencies'].append(f"Custom action: {action}")
                                        dependency_check['dependency_score'] -= 20
            
            # Check Python availability
            if dependency_check['uses_python']:
                try:
                    subprocess.run(['python', '--version'], capture_output=True, check=True)
                except:
                    try:
                        subprocess.run(['python3', '--version'], capture_output=True, check=True)
                    except:
                        dependency_check['missing_dependencies'].append('Python interpreter')
                        dependency_check['dependency_score'] -= 30
            
            # Check Node.js availability
            if dependency_check['uses_node']:
                try:
                    subprocess.run(['node', '--version'], capture_output=True, check=True)
                except:
                    dependency_check['missing_dependencies'].append('Node.js')
                    dependency_check['dependency_score'] -= 20
            
            return dependency_check
            
        except Exception as e:
            return {'error': str(e), 'dependency_score': 0}
    
    def check_workflow_schedule_validity(self, workflow_file: Path) -> Dict:
        """Check if workflow schedules are valid and when they should run"""
        try:
            with open(workflow_file, 'r', encoding='utf-8') as f:
                workflow_data = yaml.safe_load(f)
            
            schedule_info = {
                'has_schedule': False,
                'schedule_valid': False,
                'cron_expressions': [],
                'next_expected_run': None,
                'schedule_frequency': 'none'
            }
            
            if 'on' in workflow_data:
                triggers = workflow_data['on']
                if isinstance(triggers, dict) and 'schedule' in triggers:
                    schedule_info['has_schedule'] = True
                    schedule = triggers['schedule']
                    
                    if isinstance(schedule, list):
                        for item in schedule:
                            if isinstance(item, dict) and 'cron' in item:
                                cron_expr = item['cron']
                                schedule_info['cron_expressions'].append(cron_expr)
                                
                                # Try to validate cron expression
                                try:
                                    from croniter import croniter
                                    if croniter.is_valid(cron_expr):
                                        schedule_info['schedule_valid'] = True
                                        
                                        # Calculate next run
                                        cron = croniter(cron_expr, datetime.now())
                                        next_run = cron.get_next(datetime)
                                        schedule_info['next_expected_run'] = next_run.isoformat()
                                        
                                        # Estimate frequency
                                        current = datetime.now()
                                        next1 = cron.get_next(datetime)
                                        next2 = cron.get_next(datetime)
                                        interval = next2 - next1
                                        
                                        if interval.total_seconds() < 3600:  # < 1 hour
                                            schedule_info['schedule_frequency'] = 'high'
                                        elif interval.total_seconds() < 86400:  # < 1 day
                                            schedule_info['schedule_frequency'] = 'medium'
                                        else:
                                            schedule_info['schedule_frequency'] = 'low'
                                        
                                        break
                                        
                                except ImportError:
                                    # croniter not available, basic validation
                                    if ' ' in cron_expr and len(cron_expr.split()) == 5:
                                        schedule_info['schedule_valid'] = True
                                except Exception:
                                    pass
            
            return schedule_info
            
        except Exception as e:
            return {'error': str(e)}
    
    def run_workflow_validation(self, workflow_file: Path) -> Dict:
        """Run comprehensive validation of a single workflow"""
        print(f"🧪 Validating: {workflow_file.name}")
        
        validation_result = {
            'workflow_name': workflow_file.name,
            'timestamp': datetime.now().isoformat(),
            'validation_score': 0,
            'issues_found': [],
            'recommendations': []
        }
        
        # Extract commands
        commands = self.extract_key_commands_from_workflow(workflow_file)
        validation_result['commands_found'] = len(commands)
        
        # Test Python scripts
        python_test = self.test_python_scripts_in_workflow(commands)
        validation_result['python_test'] = python_test
        
        if python_test['python_scripts_found'] > 0:
            python_success_rate = python_test['python_scripts_working'] / python_test['python_scripts_found']
            validation_result['validation_score'] += python_success_rate * 30
            
            if python_success_rate < 1.0:
                validation_result['issues_found'].append(f"Some Python scripts failing: {python_test['script_test_results']}")
        else:
            validation_result['validation_score'] += 30  # No Python dependency
        
        # Check dependencies
        dep_check = self.check_workflow_dependencies(workflow_file)
        validation_result['dependency_check'] = dep_check
        validation_result['validation_score'] += (dep_check.get('dependency_score', 0) * 0.3)
        
        if dep_check.get('missing_dependencies'):
            validation_result['issues_found'].extend(dep_check['missing_dependencies'])
            validation_result['recommendations'].append("Install missing dependencies")
        
        # Check schedule
        schedule_check = self.check_workflow_schedule_validity(workflow_file)
        validation_result['schedule_check'] = schedule_check
        
        if schedule_check.get('has_schedule'):
            if schedule_check.get('schedule_valid'):
                validation_result['validation_score'] += 20
                if schedule_check.get('next_expected_run'):
                    validation_result['recommendations'].append(f"Next scheduled run: {schedule_check['next_expected_run']}")
            else:
                validation_result['issues_found'].append("Invalid cron schedule")
                validation_result['recommendations'].append("Fix cron expression in schedule")
        else:
            validation_result['validation_score'] += 10  # Manual trigger workflows
            validation_result['recommendations'].append("Workflow requires manual trigger or event-based activation")
        
        # Final assessment
        score = validation_result['validation_score']
        if score >= 80:
            validation_result['status'] = 'ready_to_run'
        elif score >= 60:
            validation_result['status'] = 'needs_minor_fixes'
        else:
            validation_result['status'] = 'needs_major_fixes'
        
        # Print summary
        status_emoji = {'ready_to_run': '✅', 'needs_minor_fixes': '⚠️', 'needs_major_fixes': '❌'}
        print(f"   {status_emoji.get(validation_result['status'], '❓')} Score: {score:.1f}/100 - {validation_result['status']}")
        
        if validation_result['issues_found']:
            print(f"   Issues: {len(validation_result['issues_found'])}")
            for issue in validation_result['issues_found'][:3]:  # Show first 3
                print(f"     • {issue}")
        
        return validation_result
    
    def run_all_workflow_validations(self) -> Dict:
        """Run validation on all workflows"""
        print("🔬 WORKFLOW RUNTIME VALIDATION SYSTEM")
        print("=" * 60)
        
        workflow_files = list(self.workflows_dir.glob('*.yml'))
        print(f"📋 Validating {len(workflow_files)} workflows for runtime readiness...")
        print()
        
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'total_workflows': len(workflow_files),
            'ready_to_run': 0,
            'needs_fixes': 0,
            'major_issues': 0,
            'workflow_results': {}
        }
        
        for workflow_file in workflow_files:
            result = self.run_workflow_validation(workflow_file)
            all_results['workflow_results'][workflow_file.name] = result
            
            if result['status'] == 'ready_to_run':
                all_results['ready_to_run'] += 1
            elif result['status'] == 'needs_minor_fixes':
                all_results['needs_fixes'] += 1
            else:
                all_results['major_issues'] += 1
        
        # Print summary
        print("\n📊 VALIDATION SUMMARY")
        print("=" * 60)
        print(f"✅ Ready to run: {all_results['ready_to_run']}/{len(workflow_files)}")
        print(f"⚠️ Need minor fixes: {all_results['needs_fixes']}/{len(workflow_files)}")
        print(f"❌ Need major fixes: {all_results['major_issues']}/{len(workflow_files)}")
        
        success_rate = all_results['ready_to_run'] / len(workflow_files) * 100
        print(f"🎯 Runtime readiness: {success_rate:.1f}%")
        
        # Show workflows that need attention
        if all_results['needs_fixes'] > 0 or all_results['major_issues'] > 0:
            print("\n🔧 WORKFLOWS NEEDING ATTENTION:")
            for wf_name, result in all_results['workflow_results'].items():
                if result['status'] != 'ready_to_run':
                    print(f"   • {wf_name} ({result['status']}) - Score: {result['validation_score']:.1f}")
                    if result['issues_found']:
                        for issue in result['issues_found'][:2]:
                            print(f"     - {issue}")
        
        return all_results
    
    def save_validation_results(self, results: Dict):
        """Save validation results"""
        results_file = Path('workflow_runtime_validation_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Validation results saved to: {results_file}")

def main():
    """Run workflow runtime validation"""
    validator = WorkflowRuntimeValidator()
    results = validator.run_all_workflow_validations()
    validator.save_validation_results(results)
    
    return results

if __name__ == "__main__":
    main()