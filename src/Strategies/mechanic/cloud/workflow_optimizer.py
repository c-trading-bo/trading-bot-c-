#!/usr/bin/env python3
"""
WORKFLOW OPTIMIZER - Applies optimizations to workflows
"""

import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List

class WorkflowOptimizer:
    def __init__(self):
        self.cloud_path = Path('Intelligence/mechanic/cloud')
        self.workflow_path = Path('.github/workflows')
        self.optimizations_applied = []
        
    def optimize_all_workflows(self):
        """Apply optimizations to all workflows"""
        print("\n⚡ OPTIMIZING WORKFLOWS...")
        
        optimization_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'optimizations_applied': [],
            'minutes_saved_estimate': 0,
            'files_modified': []
        }
        
        if not self.workflow_path.exists():
            print("❌ No .github/workflows directory found")
            return optimization_results
            
        workflow_files = list(self.workflow_path.glob('*.yml')) + list(self.workflow_path.glob('*.yaml'))
        
        for wf_file in workflow_files:
            minutes_saved = self.optimize_workflow(wf_file)
            if minutes_saved > 0:
                optimization_results['files_modified'].append(wf_file.name)
                optimization_results['minutes_saved_estimate'] += minutes_saved
        
        optimization_results['optimizations_applied'] = self.optimizations_applied
        
        # Save results
        output_path = self.cloud_path / 'database' / 'optimizations.json'
        self.save_json(output_path, optimization_results)
        
        self.print_optimization_summary(optimization_results)
        
        # Set GitHub output for changes
        if optimization_results['files_modified']:
            self.set_github_output('changes_made', 'true')
        
        return optimization_results
    
    def optimize_workflow(self, wf_file: Path) -> int:
        """Optimize a single workflow file"""
        try:
            # Try multiple encodings to handle special characters
            workflow = None
            for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
                try:
                    with open(wf_file, 'r', encoding=encoding) as f:
                        workflow = yaml.safe_load(f)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if not workflow:
                return 0
            
            original_content = yaml.dump(workflow, default_flow_style=False)
            minutes_saved = 0
            
            # Apply optimizations
            minutes_saved += self.add_caching(workflow, wf_file.name)
            minutes_saved += self.optimize_checkout(workflow, wf_file.name)
            minutes_saved += self.set_timeouts(workflow, wf_file.name)
            minutes_saved += self.add_conditional_execution(workflow, wf_file.name)
            
            # Save if modified
            new_content = yaml.dump(workflow, default_flow_style=False)
            if new_content != original_content:
                # Use the same encoding that worked for reading
                for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
                    try:
                        with open(wf_file, 'w', encoding=encoding) as f:
                            yaml.dump(workflow, f, default_flow_style=False, sort_keys=False)
                        break
                    except (UnicodeEncodeError, UnicodeError):
                        continue
                
                print(f"    ✅ Optimized {wf_file.name}")
                return minutes_saved
            
            return 0
            
        except Exception as e:
            print(f"    ❌ Failed to optimize {wf_file.name}: {e}")
            return 0
    
    def add_caching(self, workflow: Dict, wf_name: str) -> int:
        """Add caching to workflow"""
        minutes_saved = 0
        
        for job_name, job in workflow.get('jobs', {}).items():
            steps = job.get('steps', [])
            
            # Add cache to setup-python steps
            for step in steps:
                if isinstance(step, dict) and 'setup-python' in step.get('uses', ''):
                    if 'with' not in step:
                        step['with'] = {}
                    if 'cache' not in step['with']:
                        step['with']['cache'] = 'pip'
                        minutes_saved += 2
                        self.optimizations_applied.append(f"{wf_name}: Added pip cache to {job_name}")
            
            # Add general dependency cache for jobs with many steps
            has_cache = any('cache' in step.get('uses', '') for step in steps if isinstance(step, dict))
            if not has_cache and len(steps) > 5:
                cache_step = {
                    'name': 'Cache dependencies',
                    'uses': 'actions/cache@v3',
                    'with': {
                        'path': '~/.cache/pip',
                        'key': "${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}",
                        'restore-keys': "${{ runner.os }}-pip-"
                    }
                }
                # Insert cache after checkout (usually step 0 or 1)
                insert_position = min(2, len(steps))
                steps.insert(insert_position, cache_step)
                minutes_saved += 3
                self.optimizations_applied.append(f"{wf_name}: Added dependency cache to {job_name}")
        
        return minutes_saved
    
    def optimize_checkout(self, workflow: Dict, wf_name: str) -> int:
        """Optimize checkout steps for faster cloning"""
        minutes_saved = 0
        
        for job_name, job in workflow.get('jobs', {}).items():
            for step in job.get('steps', []):
                if isinstance(step, dict) and 'checkout' in step.get('uses', ''):
                    if 'with' not in step:
                        step['with'] = {}
                    if 'fetch-depth' not in step['with']:
                        step['with']['fetch-depth'] = 1  # Shallow clone
                        minutes_saved += 1
                        self.optimizations_applied.append(f"{wf_name}: Added shallow clone to {job_name}")
        
        return minutes_saved
    
    def set_timeouts(self, workflow: Dict, wf_name: str) -> int:
        """Set appropriate timeouts for jobs"""
        minutes_saved = 0
        
        for job_name, job in workflow.get('jobs', {}).items():
            steps_count = len(job.get('steps', []))
            
            if 'timeout-minutes' not in job:
                # Estimate reasonable timeout based on steps
                estimated_timeout = min(max(steps_count * 3, 10), 45)
                job['timeout-minutes'] = estimated_timeout
                minutes_saved += 5  # Prevents hanging jobs
                self.optimizations_applied.append(f"{wf_name}: Set {estimated_timeout}m timeout for {job_name}")
            
            elif job.get('timeout-minutes', 0) > 60:
                # Reduce excessive timeouts
                job['timeout-minutes'] = 45
                minutes_saved += 10
                self.optimizations_applied.append(f"{wf_name}: Reduced timeout for {job_name}")
        
        return minutes_saved
    
    def add_conditional_execution(self, workflow: Dict, wf_name: str) -> int:
        """Add conditional execution to reduce unnecessary runs"""
        minutes_saved = 0
        
        triggers = workflow.get('on', {})
        
        # Add path filters to push triggers
        if 'push' in triggers and not isinstance(triggers['push'], dict):
            triggers['push'] = {
                'branches': ['main'],
                'paths-ignore': ['**.md', 'docs/**', '.gitignore', '*.txt']
            }
            minutes_saved += 10  # Significant savings from avoiding doc-only runs
            self.optimizations_applied.append(f"{wf_name}: Added path filters to push trigger")
        
        # Add branch filters to pull_request if missing
        if 'pull_request' in triggers and not isinstance(triggers['pull_request'], dict):
            triggers['pull_request'] = {
                'branches': ['main']
            }
            minutes_saved += 5
            self.optimizations_applied.append(f"{wf_name}: Added branch filter to PR trigger")
        
        return minutes_saved
    
    def print_optimization_summary(self, results: Dict):
        """Print optimization summary"""
        print(f"\n⚡ OPTIMIZATION SUMMARY")
        print(f"   Files modified: {len(results['files_modified'])}")
        print(f"   Estimated minutes saved/month: {results['minutes_saved_estimate']}")
        print(f"   Optimizations applied: {len(results['optimizations_applied'])}")
        
        if results['optimizations_applied']:
            print("   Recent optimizations:")
            for opt in results['optimizations_applied'][-5:]:  # Show last 5
                print(f"     • {opt}")
    
    def set_github_output(self, name: str, value: str):
        """Set GitHub Actions output"""
        try:
            if os.environ.get('GITHUB_OUTPUT'):
                with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                    f.write(f"{name}={value}\n")
        except Exception as e:
            print(f"⚠️ Failed to set GitHub output {name}: {e}")
    
    def save_json(self, path: Path, data: Dict):
        """Save JSON data"""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Failed to save {path}: {e}")

if __name__ == "__main__":
    optimizer = WorkflowOptimizer()
    optimizer.optimize_all_workflows()
