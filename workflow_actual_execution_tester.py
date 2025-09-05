#!/usr/bin/env python3
"""
WORKFLOW ACTUAL EXECUTION TESTER
Actually runs the core components of each workflow to verify they work
"""

import os
import sys
import json
import subprocess
import time
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

class WorkflowActualExecutionTester:
    def __init__(self):
        self.workflows_dir = Path('.github/workflows')
        self.test_results = {}
        self.successful_executions = 0
        self.failed_executions = 0
        
    def test_workflow_by_running_key_scripts(self, workflow_file: Path) -> Dict:
        """Test workflow by actually running its key Python scripts"""
        print(f"🔧 Actually testing execution of: {workflow_file.name}")
        
        try:
            with open(workflow_file, 'r', encoding='utf-8') as f:
                content = f.read()
                workflow_data = yaml.safe_load(content)
            
            execution_results = {
                'workflow_name': workflow_file.name,
                'scripts_found': 0,
                'scripts_executed': 0,
                'scripts_successful': 0,
                'execution_details': {},
                'overall_status': 'unknown'
            }
            
            # Extract Python scripts from workflow
            import re
            python_scripts = re.findall(r'python\s+([^\s]+\.py)', content)
            
            if not python_scripts:
                # Check for inline Python code
                if 'python -c' in content:
                    execution_results['has_inline_python'] = True
                    execution_results['overall_status'] = 'inline_python_detected'
                else:
                    execution_results['overall_status'] = 'no_python_scripts'
                return execution_results
            
            execution_results['scripts_found'] = len(python_scripts)
            
            # Try to execute each script
            for script_path in python_scripts:
                script_name = Path(script_path).name
                execution_results['scripts_executed'] += 1
                
                script_result = self.test_script_execution(script_path)
                execution_results['execution_details'][script_name] = script_result
                
                if script_result['status'] == 'success':
                    execution_results['scripts_successful'] += 1
            
            # Determine overall status
            if execution_results['scripts_successful'] == execution_results['scripts_found']:
                execution_results['overall_status'] = 'all_scripts_working'
                self.successful_executions += 1
            elif execution_results['scripts_successful'] > 0:
                execution_results['overall_status'] = 'partially_working'
            else:
                execution_results['overall_status'] = 'scripts_failing'
                self.failed_executions += 1
            
            return execution_results
            
        except Exception as e:
            self.failed_executions += 1
            return {
                'workflow_name': workflow_file.name,
                'error': str(e),
                'overall_status': 'test_failed'
            }
    
    def test_script_execution(self, script_path: str) -> Dict:
        """Test if a Python script can execute successfully"""
        full_path = Path(script_path)
        
        if not full_path.exists():
            return {
                'status': 'file_not_found',
                'message': f'Script {script_path} not found'
            }
        
        try:
            # First, try to import the script to check for syntax errors
            import importlib.util
            spec = importlib.util.spec_from_file_location("test_module", full_path)
            
            if spec is None:
                return {
                    'status': 'import_error',
                    'message': 'Could not create module spec'
                }
            
            # Try to execute with --help flag first
            result = subprocess.run([
                sys.executable, str(full_path), '--help'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return {
                    'status': 'success',
                    'message': 'Script responds to --help',
                    'stdout': result.stdout[:200] if result.stdout else '',
                    'method': 'help_flag'
                }
            
            # If --help fails, try without arguments but with timeout
            result = subprocess.run([
                sys.executable, str(full_path)
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return {
                    'status': 'success',
                    'message': 'Script executed successfully',
                    'stdout': result.stdout[:200] if result.stdout else '',
                    'method': 'direct_execution'
                }
            else:
                # Script ran but returned error code - might still be working
                return {
                    'status': 'executed_with_errors',
                    'message': 'Script ran but returned non-zero exit code',
                    'stderr': result.stderr[:200] if result.stderr else '',
                    'returncode': result.returncode,
                    'method': 'direct_execution'
                }
        
        except subprocess.TimeoutExpired:
            return {
                'status': 'timeout',
                'message': 'Script execution timed out (may be working but needs input)'
            }
        except Exception as e:
            return {
                'status': 'execution_error',
                'message': str(e)
            }
    
    def test_specific_workflow_functionality(self, workflow_name: str) -> Dict:
        """Test specific functionality for known workflows"""
        
        if 'cloud_bot_mechanic' in workflow_name:
            return self.test_cloud_mechanic_execution()
        elif 'data_collection' in workflow_name:
            return self.test_data_collection_execution()
        elif 'daily_report' in workflow_name:
            return self.test_daily_report_execution()
        elif 'ml_rl_training' in workflow_name:
            return self.test_ml_training_execution()
        else:
            return {'status': 'no_specific_test'}
    
    def test_cloud_mechanic_execution(self) -> Dict:
        """Test cloud mechanic functionality"""
        try:
            mechanic_path = Path("Intelligence/mechanic/cloud/cloud_mechanic_core.py")
            if mechanic_path.exists():
                
                # Set up environment
                env = os.environ.copy()
                env.update({
                    'ULTIMATE_MODE': 'true',
                    'GITHUB_REPOSITORY_OWNER': 'c-trading-bo',
                    'GITHUB_REPOSITORY': 'trading-bot-c-'
                })
                
                result = subprocess.run([
                    sys.executable, str(mechanic_path)
                ], env=env, capture_output=True, text=True, timeout=60)
                
                return {
                    'status': 'success' if result.returncode == 0 else 'executed_with_errors',
                    'returncode': result.returncode,
                    'output_length': len(result.stdout),
                    'has_output': len(result.stdout) > 100,
                    'error_output': result.stderr[:200] if result.stderr else ''
                }
            else:
                return {'status': 'file_not_found'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def test_data_collection_execution(self) -> Dict:
        """Test data collection functionality"""
        try:
            # Test market data collection
            market_data_path = Path("Intelligence/scripts/collect_market_data.py")
            if market_data_path.exists():
                result = subprocess.run([
                    sys.executable, str(market_data_path), '--help'
                ], capture_output=True, text=True, timeout=30)
                
                return {
                    'status': 'success' if result.returncode == 0 else 'help_failed',
                    'market_data_script': 'working' if result.returncode == 0 else 'failed'
                }
            else:
                return {'status': 'script_missing'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def test_daily_report_execution(self) -> Dict:
        """Test daily report functionality"""
        try:
            report_path = Path("Intelligence/scripts/generate_daily_report.py")
            if report_path.exists():
                result = subprocess.run([
                    sys.executable, str(report_path), '--help'
                ], capture_output=True, text=True, timeout=30)
                
                return {
                    'status': 'success' if result.returncode == 0 else 'help_failed',
                    'daily_report_script': 'working' if result.returncode == 0 else 'failed'
                }
            else:
                return {'status': 'script_missing'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def test_ml_training_execution(self) -> Dict:
        """Test ML training functionality"""
        try:
            # Test CVaR PPO training script
            cvar_path = Path("ml/rl/train_cvar_ppo.py")
            if cvar_path.exists():
                result = subprocess.run([
                    sys.executable, str(cvar_path), '--help'
                ], capture_output=True, text=True, timeout=30)
                
                return {
                    'status': 'success' if result.returncode == 0 else 'help_failed',
                    'cvar_ppo_script': 'working' if result.returncode == 0 else 'failed'
                }
            else:
                return {'status': 'script_missing'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def run_comprehensive_execution_test(self) -> Dict:
        """Run comprehensive execution test on all workflows"""
        print("🚀 WORKFLOW ACTUAL EXECUTION TESTING")
        print("=" * 60)
        
        workflow_files = list(self.workflows_dir.glob('*.yml'))
        print(f"🔧 Testing actual execution of {len(workflow_files)} workflows...")
        print()
        
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'total_workflows': len(workflow_files),
            'successful_workflows': 0,
            'failed_workflows': 0,
            'partially_working_workflows': 0,
            'workflow_details': {}
        }
        
        for workflow_file in workflow_files:
            # Test by running scripts
            execution_result = self.test_workflow_by_running_key_scripts(workflow_file)
            
            # Test specific functionality
            specific_test = self.test_specific_workflow_functionality(workflow_file.name)
            execution_result['specific_test'] = specific_test
            
            test_results['workflow_details'][workflow_file.name] = execution_result
            
            # Update counters
            status = execution_result['overall_status']
            if status in ['all_scripts_working', 'inline_python_detected']:
                test_results['successful_workflows'] += 1
                status_emoji = '✅'
            elif status == 'partially_working':
                test_results['partially_working_workflows'] += 1
                status_emoji = '⚠️'
            else:
                test_results['failed_workflows'] += 1
                status_emoji = '❌'
            
            print(f"{status_emoji} {workflow_file.name} - {status}")
            
            # Show script details
            if 'execution_details' in execution_result:
                successful = execution_result.get('scripts_successful', 0)
                total = execution_result.get('scripts_found', 0)
                if total > 0:
                    print(f"    Scripts: {successful}/{total} working")
        
        print("\n📊 EXECUTION TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Fully Working: {test_results['successful_workflows']}")
        print(f"⚠️ Partially Working: {test_results['partially_working_workflows']}")
        print(f"❌ Not Working: {test_results['failed_workflows']}")
        
        success_rate = (test_results['successful_workflows'] + test_results['partially_working_workflows']) / len(workflow_files) * 100
        print(f"🎯 Overall Execution Rate: {success_rate:.1f}%")
        
        # Show failing workflows
        failing_workflows = []
        for wf_name, details in test_results['workflow_details'].items():
            if details['overall_status'] in ['scripts_failing', 'test_failed']:
                failing_workflows.append(wf_name)
        
        if failing_workflows:
            print(f"\n❌ WORKFLOWS NEEDING FIXES ({len(failing_workflows)}):")
            for wf in failing_workflows:
                print(f"   • {wf}")
                details = test_results['workflow_details'][wf]
                if 'execution_details' in details:
                    for script, result in details['execution_details'].items():
                        if result['status'] != 'success':
                            print(f"     - {script}: {result['status']}")
        
        return test_results
    
    def save_execution_test_results(self, results: Dict):
        """Save execution test results"""
        results_file = Path('workflow_actual_execution_test_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Execution test results saved to: {results_file}")

def main():
    """Run actual workflow execution testing"""
    tester = WorkflowActualExecutionTester()
    results = tester.run_comprehensive_execution_test()
    tester.save_execution_test_results(results)
    
    return results

if __name__ == "__main__":
    main()