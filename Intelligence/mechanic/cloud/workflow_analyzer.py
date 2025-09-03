#!/usr/bin/env python3
"""
WORKFLOW ANALYZER - Detailed analysis of individual workflows
"""

import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List

class WorkflowAnalyzer:
    def __init__(self):
        self.cloud_path = Path('Intelligence/mechanic/cloud')
        self.workflow_path = Path('.github/workflows')
        
    def analyze_workflows(self):
        """Perform detailed analysis of all workflows"""
        print("\n🔬 DETAILED WORKFLOW ANALYSIS...")
        
        analysis_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'workflows': {},
            'summary': {
                'total_jobs': 0,
                'total_steps': 0,
                'optimization_hints': []
            }
        }
        
        if not self.workflow_path.exists():
            print("❌ No .github/workflows directory found")
            return analysis_results
            
        workflow_files = list(self.workflow_path.glob('*.yml')) + list(self.workflow_path.glob('*.yaml'))
        
        for wf_file in workflow_files:
            result = self.analyze_single_workflow(wf_file)
            analysis_results['workflows'][wf_file.name] = result
            
            # Update summary
            analysis_results['summary']['total_jobs'] += result.get('job_count', 0)
            analysis_results['summary']['total_steps'] += result.get('step_count', 0)
        
        # Generate optimization hints
        self.generate_optimization_hints(analysis_results)
        
        # Save results
        output_path = self.cloud_path / 'database' / 'detailed_analysis.json'
        self.save_json(output_path, analysis_results)
        
        self.print_analysis_summary(analysis_results)
        return analysis_results
    
    def analyze_single_workflow(self, wf_file: Path) -> Dict:
        """Analyze a single workflow file"""
        try:
            with open(wf_file, 'r') as f:
                workflow = yaml.safe_load(f)
            
            if not workflow:
                return {'error': 'Empty workflow file'}
            
            result = {
                'name': workflow.get('name', wf_file.name),
                'triggers': list(workflow.get('on', {}).keys()),
                'job_count': len(workflow.get('jobs', {})),
                'step_count': 0,
                'jobs': {},
                'issues': [],
                'suggestions': []
            }
            
            # Analyze each job
            for job_name, job_config in workflow.get('jobs', {}).items():
                job_analysis = self.analyze_job(job_name, job_config)
                result['jobs'][job_name] = job_analysis
                result['step_count'] += job_analysis.get('step_count', 0)
            
            # Check for common issues
            self.check_workflow_issues(workflow, result)
            
            return result
            
        except Exception as e:
            return {'error': f'Failed to analyze: {e}'}
    
    def analyze_job(self, job_name: str, job_config: Dict) -> Dict:
        """Analyze a single job"""
        steps = job_config.get('steps', [])
        
        job_analysis = {
            'step_count': len(steps),
            'runs_on': job_config.get('runs-on', 'ubuntu-latest'),
            'timeout': job_config.get('timeout-minutes', 'default'),
            'has_cache': False,
            'has_artifacts': False,
            'step_types': []
        }
        
        # Analyze steps
        for step in steps:
            if isinstance(step, dict):
                # Check for caching
                if 'cache' in step.get('uses', '') or 'cache' in str(step.get('with', {})):
                    job_analysis['has_cache'] = True
                
                # Check for artifacts
                if 'artifact' in step.get('uses', ''):
                    job_analysis['has_artifacts'] = True
                
                # Categorize step type
                if step.get('uses'):
                    job_analysis['step_types'].append('action')
                elif step.get('run'):
                    job_analysis['step_types'].append('script')
        
        return job_analysis
    
    def check_workflow_issues(self, workflow: Dict, result: Dict):
        """Check for common workflow issues"""
        # Check for missing timeouts
        for job_name, job in workflow.get('jobs', {}).items():
            if 'timeout-minutes' not in job:
                result['suggestions'].append(f"Add timeout to job '{job_name}'")
        
        # Check for matrix opportunities
        jobs = workflow.get('jobs', {})
        if len(jobs) > 2:
            similar_runners = {}
            for job_name, job in jobs.items():
                runner = job.get('runs-on', 'ubuntu-latest')
                if runner not in similar_runners:
                    similar_runners[runner] = []
                similar_runners[runner].append(job_name)
            
            for runner, job_list in similar_runners.items():
                if len(job_list) > 2:
                    result['suggestions'].append(f"Consider matrix strategy for {runner} jobs: {job_list}")
    
    def generate_optimization_hints(self, analysis: Dict):
        """Generate overall optimization hints"""
        hints = []
        
        total_jobs = analysis['summary']['total_jobs']
        total_steps = analysis['summary']['total_steps']
        
        if total_steps > 100:
            hints.append("High step count detected - consider job consolidation")
        
        # Check for workflows without caching
        workflows_without_cache = 0
        for wf_name, wf_data in analysis['workflows'].items():
            if isinstance(wf_data, dict) and 'jobs' in wf_data:
                has_any_cache = any(job.get('has_cache', False) for job in wf_data['jobs'].values())
                if not has_any_cache:
                    workflows_without_cache += 1
        
        if workflows_without_cache > 0:
            hints.append(f"{workflows_without_cache} workflows could benefit from caching")
        
        analysis['summary']['optimization_hints'] = hints
    
    def print_analysis_summary(self, analysis: Dict):
        """Print analysis summary"""
        print(f"\n📈 DETAILED ANALYSIS SUMMARY")
        print(f"   Workflows analyzed: {len(analysis['workflows'])}")
        print(f"   Total jobs: {analysis['summary']['total_jobs']}")
        print(f"   Total steps: {analysis['summary']['total_steps']}")
        
        if analysis['summary']['optimization_hints']:
            print(f"   Optimization hints:")
            for hint in analysis['summary']['optimization_hints']:
                print(f"     • {hint}")
    
    def save_json(self, path: Path, data: Dict):
        """Save JSON data"""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Failed to save {path}: {e}")

if __name__ == "__main__":
    analyzer = WorkflowAnalyzer()
    analyzer.analyze_workflows()
