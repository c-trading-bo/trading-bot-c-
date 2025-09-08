#!/usr/bin/env python3
"""
WORKFLOW LEARNER - Intelligent Workflow Analysis and Optimization
Learns from workflow patterns and generates optimizations
"""

import os
import sys
import json
import yaml
import re
import requests
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import traceback

class WorkflowLearner:
    """Intelligent workflow analysis and learning system"""
    
    def __init__(self):
        self.workflows = {}
        self.step_patterns = {}
        self.optimization_strategies = {}
        self.failure_patterns = {}
        self.success_metrics = {}
        self.github_token = os.environ.get('GITHUB_TOKEN', '')
        
        self.initialize_patterns()
    
    def initialize_patterns(self):
        """Initialize known step patterns and their optimizations"""
        self.step_patterns = {
            # Checkout patterns
            'actions/checkout': {
                'type': 'checkout',
                'optimization': 'shallow-clone',
                'prep': ['pre-fetch-refs', 'cache-repo'],
                'cache': True,
                'time_avg': 30000
            },
            
            # Node setup patterns
            'actions/setup-node': {
                'type': 'runtime-setup',
                'optimization': 'pre-warm-runtime',
                'prep': ['download-node-binary', 'setup-npm-cache'],
                'cache': True,
                'time_avg': 45000
            },
            
            # Python setup patterns
            'actions/setup-python': {
                'type': 'runtime-setup',
                'optimization': 'pre-install-python',
                'prep': ['download-python', 'setup-pip-cache'],
                'cache': True,
                'time_avg': 60000
            },
            
            # Dependency installation patterns
            'npm install': {
                'type': 'dependency-install',
                'optimization': 'pre-bundle-deps',
                'prep': ['resolve-deps', 'download-packages', 'create-lockfile'],
                'cache': True,
                'time_avg': 120000
            },
            
            'npm ci': {
                'type': 'dependency-install',
                'optimization': 'use-frozen-lockfile',
                'prep': ['verify-lockfile', 'restore-node-modules'],
                'cache': True,
                'time_avg': 90000
            },
            
            'pip install': {
                'type': 'dependency-install',
                'optimization': 'pre-wheel-packages',
                'prep': ['download-wheels', 'compile-c-extensions'],
                'cache': True,
                'time_avg': 150000
            },
            
            # Test patterns
            'npm test': {
                'type': 'testing',
                'optimization': 'parallel-test-execution',
                'prep': ['identify-changed-files', 'split-test-suites'],
                'cache': False,
                'parallel': True,
                'time_avg': 180000
            },
            
            'pytest': {
                'type': 'testing',
                'optimization': 'pytest-xdist',
                'prep': ['collect-tests', 'setup-workers'],
                'cache': False,
                'parallel': True,
                'time_avg': 120000
            },
            
            # Build patterns
            'npm run build': {
                'type': 'build',
                'optimization': 'incremental-build',
                'prep': ['analyze-deps-graph', 'identify-unchanged'],
                'cache': True,
                'time_avg': 90000
            },
            
            'tsc': {
                'type': 'build',
                'optimization': 'typescript-incremental',
                'prep': ['parse-tsconfig', 'cache-type-definitions'],
                'cache': True,
                'time_avg': 60000
            },
            
            'dotnet build': {
                'type': 'build',
                'optimization': 'dotnet-incremental',
                'prep': ['restore-packages', 'cache-nuget'],
                'cache': True,
                'time_avg': 75000
            },
            
            # Docker patterns
            'docker build': {
                'type': 'containerization',
                'optimization': 'layer-caching',
                'prep': ['pull-base-images', 'cache-layers'],
                'cache': True,
                'time_avg': 240000
            },
            
            # Deploy patterns
            'deploy': {
                'type': 'deployment',
                'optimization': 'blue-green-deploy',
                'prep': ['warm-cdn', 'pre-scale-servers'],
                'cache': False,
                'time_avg': 60000
            }
        }
    
    async def learn_from_workflow_file(self, repo_path: str, workflow_file: str) -> Optional[Dict]:
        """Learn from a workflow file"""
        workflow_path = Path(repo_path) / '.github' / 'workflows' / workflow_file
        
        if not workflow_path.exists():
            # Try to fetch from GitHub
            return await self.fetch_workflow_from_github(repo_path, workflow_file)
        
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.parse_and_learn_workflow(content, workflow_file)
        except Exception as e:
            print(f"❌ Error learning from {workflow_file}: {e}")
            return None
    
    def parse_and_learn_workflow(self, yaml_content: str, file_name: str) -> Dict:
        """Parse and learn from workflow YAML content"""
        try:
            workflow = self.parse_yaml(yaml_content)
        except Exception as e:
            print(f"❌ Error parsing YAML for {file_name}: {e}")
            return {}
        
        learned = {
            'name': workflow.get('name', file_name),
            'triggers': self.analyze_triggers(workflow.get('on')),
            'jobs': {},
            'estimated_time': 0,
            'optimizations': [],
            'dependencies': [],
            'critical_path': []
        }
        
        # Analyze each job
        jobs = workflow.get('jobs', {})
        for job_id, job in jobs.items():
            job_analysis = {
                'id': job_id,
                'runs_on': job.get('runs-on'),
                'needs': job.get('needs', []),
                'strategy': job.get('strategy'),
                'steps': [],
                'can_parallelize': not job.get('needs') or len(job.get('needs', [])) == 0,
                'estimated_time': 0
            }
            
            # Analyze each step
            steps = job.get('steps', [])
            for i, step in enumerate(steps):
                step_analysis = self.analyze_step(step, i, steps)
                job_analysis['steps'].append(step_analysis)
                job_analysis['estimated_time'] += step_analysis['estimated_time']
            
            learned['jobs'][job_id] = job_analysis
            learned['estimated_time'] = max(learned['estimated_time'], job_analysis['estimated_time'])
        
        # Calculate critical path
        learned['critical_path'] = self.calculate_critical_path(learned['jobs'])
        
        # Generate optimizations
        learned['optimizations'] = self.generate_optimizations(learned)
        
        return learned
    
    def analyze_step(self, step: Dict, index: int, all_steps: List[Dict]) -> Dict:
        """Analyze a workflow step for optimization opportunities"""
        analysis = {
            'name': step.get('name', f'Step {index + 1}'),
            'index': index,
            'uses': step.get('uses'),
            'run': step.get('run'),
            'with': step.get('with', {}),
            'env': step.get('env', {}),
            'if': step.get('if'),
            'continue_on_error': step.get('continue-on-error', False),
            'timeout_minutes': step.get('timeout-minutes', 360),
            'optimization': None,
            'estimated_time': 30000,  # Default 30 seconds
            'can_cache': False,
            'can_parallelize': False,
            'can_skip': False,
            'prep_work': [],
            'dependencies': []
        }
        
        # Match against known patterns
        if step.get('uses'):
            action = step['uses'].split('@')[0]
            if action in self.step_patterns:
                pattern = self.step_patterns[action]
                analysis['optimization'] = pattern['optimization']
                analysis['prep_work'] = pattern['prep']
                analysis['can_cache'] = pattern['cache']
                analysis['can_parallelize'] = pattern.get('parallel', False)
                analysis['estimated_time'] = pattern['time_avg']
        
        if step.get('run'):
            # Analyze shell commands
            commands = step['run'].split('\n')
            commands = [cmd.strip() for cmd in commands if cmd.strip()]
            
            for cmd in commands:
                # Check for known commands
                for pattern, config in self.step_patterns.items():
                    if pattern in cmd:
                        analysis['optimization'] = config['optimization']
                        analysis['prep_work'] = config['prep']
                        analysis['can_cache'] = config['cache']
                        analysis['can_parallelize'] = config.get('parallel', False)
                        analysis['estimated_time'] = max(analysis['estimated_time'], config['time_avg'])
                        break
                
                # Special checks
                if any(x in cmd for x in ['npm', 'yarn']):
                    analysis['dependencies'].append('node')
                if any(x in cmd for x in ['pip', 'python']):
                    analysis['dependencies'].append('python')
                if 'docker' in cmd:
                    analysis['dependencies'].append('docker')
                if any(x in cmd for x in ['test', 'spec']):
                    analysis['can_parallelize'] = True
                if any(x in cmd for x in ['cache', 'restore']):
                    analysis['can_cache'] = True
        
        # Check if step can be skipped
        if step.get('if'):
            analysis['can_skip'] = True
        
        # Check dependencies on previous steps
        if index > 0 and not analysis['can_parallelize']:
            prev_step = all_steps[index - 1]
            if not prev_step.get('if'):
                analysis['dependencies'].append(f'step-{index - 1}')
        
        return analysis
    
    def analyze_triggers(self, triggers: Any) -> Dict:
        """Analyze workflow triggers"""
        analysis = {
            'types': [],
            'schedule': None,
            'branches': [],
            'paths': [],
            'manual': False
        }
        
        if not triggers:
            return analysis
        
        if isinstance(triggers, str):
            analysis['types'].append(triggers)
        elif isinstance(triggers, list):
            analysis['types'] = triggers
        elif isinstance(triggers, dict):
            for trigger, config in triggers.items():
                analysis['types'].append(trigger)
                
                if trigger == 'schedule':
                    if isinstance(config, list) and config:
                        analysis['schedule'] = config[0].get('cron') if isinstance(config[0], dict) else None
                elif trigger == 'push' and isinstance(config, dict) and config.get('branches'):
                    analysis['branches'] = config['branches']
                elif trigger == 'workflow_dispatch':
                    analysis['manual'] = True
                
                if isinstance(config, dict) and config.get('paths'):
                    analysis['paths'] = config['paths']
        
        return analysis
    
    def calculate_critical_path(self, jobs: Dict) -> List[str]:
        """Find the longest path through job dependencies"""
        path = []
        visited = set()
        
        def find_path(job_id: str, current_path: List[str] = None) -> List[str]:
            if current_path is None:
                current_path = []
            
            if job_id in visited:
                return current_path
            
            visited.add(job_id)
            job = jobs.get(job_id)
            if not job:
                return current_path
            
            current_path = current_path + [job_id]
            
            # Find jobs that depend on this one
            for other_id, other_job in jobs.items():
                needs = other_job.get('needs', [])
                if isinstance(needs, str):
                    needs = [needs]
                if job_id in needs:
                    current_path = find_path(other_id, current_path)
            
            return current_path
        
        # Start from jobs with no dependencies
        for job_id, job in jobs.items():
            needs = job.get('needs', [])
            if not needs or len(needs) == 0:
                job_path = find_path(job_id)
                if len(job_path) > len(path):
                    path = job_path
        
        return path
    
    def generate_optimizations(self, workflow: Dict) -> List[Dict]:
        """Generate optimization recommendations for a workflow"""
        optimizations = []
        jobs = workflow.get('jobs', {})
        
        # Job-level optimizations
        parallel_jobs = []
        for job_id, job in jobs.items():
            if job.get('can_parallelize', False):
                parallel_jobs.append(job_id)
        
        if len(parallel_jobs) > 1:
            optimizations.append({
                'type': 'parallelize-jobs',
                'description': f'Run jobs in parallel: {", ".join(parallel_jobs)}',
                'time_reduction': 0.5 if len(parallel_jobs) > 1 else 0,
                'implementation': f'''
jobs:
  {chr(10).join([f'{j}:' + chr(10) + '    needs: []' for j in parallel_jobs])}
                '''
            })
        
        # Step-level optimizations
        for job_id, job in jobs.items():
            steps = job.get('steps', [])
            
            # Cache optimizations
            cacheable_steps = [s for s in steps if s.get('can_cache', False)]
            if cacheable_steps:
                optimizations.append({
                    'type': 'add-caching',
                    'job': job_id,
                    'description': f'Add caching for {len(cacheable_steps)} steps',
                    'time_reduction': 0.3,
                    'implementation': '''
      - uses: actions/cache@v3
        with:
          path: |
            ~/.npm
            ~/.cache
            node_modules
          key: ${{ runner.os }}-${{ hashFiles('**/package-lock.json') }}
                    '''
                })
            
            # Parallel test execution
            test_steps = [s for s in steps if s.get('can_parallelize', False) and 
                         s.get('name', '').lower().find('test') != -1]
            if test_steps:
                optimizations.append({
                    'type': 'parallel-tests',
                    'job': job_id,
                    'description': 'Run tests in parallel',
                    'time_reduction': 0.4,
                    'implementation': '''
      - name: Run Tests in Parallel
        run: |
          npm test -- --parallel --maxWorkers=4
                    '''
                })
            
            # Skip unnecessary steps
            skippable_steps = [s for s in steps if s.get('can_skip', False)]
            if skippable_steps:
                optimizations.append({
                    'type': 'conditional-steps',
                    'job': job_id,
                    'description': f'Add conditions to skip {len(skippable_steps)} steps when not needed',
                    'time_reduction': 0.2,
                    'implementation': '''
      if: github.event_name == 'push' && contains(github.event.head_commit.message, '[skip-tests]') != true
                    '''
                })
        
        # Matrix build optimization
        if len(jobs) > 3:
            optimizations.append({
                'type': 'matrix-strategy',
                'description': 'Use matrix strategy to reduce duplicate job definitions',
                'time_reduction': 0.1,
                'implementation': '''
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        version: [16, 18, 20]
                '''
            })
        
        return optimizations
    
    def parse_yaml(self, content: str) -> Dict:
        """Simple YAML parser for workflows"""
        try:
            return yaml.safe_load(content)
        except yaml.YAMLError:
            # Fallback to simple parsing
            return self.simple_yaml_parse(content)
    
    def simple_yaml_parse(self, content: str) -> Dict:
        """Fallback simple YAML parser"""
        result = {}
        lines = content.split('\n')
        current_path = []
        current_obj = result
        stack = [result]
        
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            
            indent = len(line) - len(line.lstrip())
            depth = indent // 2
            
            # Handle key-value pairs
            if ':' in stripped:
                key, _, value = stripped.partition(':')
                key = key.strip()
                value = value.strip()
                
                # Adjust stack based on indentation
                while len(stack) > depth + 1:
                    stack.pop()
                    current_path.pop()
                
                current_obj = stack[-1]
                
                if value:
                    # Simple value
                    current_obj[key] = value.strip('\'"')
                else:
                    # New object or array
                    current_obj[key] = {}
                    stack.append(current_obj[key])
                    current_path.append(key)
        
        return result
    
    async def fetch_workflow_from_github(self, repo_info: str, workflow_file: str) -> Optional[Dict]:
        """Fetch workflow from GitHub API"""
        if not self.github_token:
            return None
        
        # Extract owner/repo from repo_info
        if '/' in repo_info:
            owner, repo = repo_info.split('/', 1)
        else:
            # Assume it's just the repo name and use environment
            owner = os.environ.get('GITHUB_REPOSITORY_OWNER', '')
            repo = repo_info
        
        url = f'https://api.github.com/repos/{owner}/{repo}/contents/.github/workflows/{workflow_file}'
        
        try:
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            content = base64.b64decode(data['content']).decode('utf-8')
            return self.parse_and_learn_workflow(content, workflow_file)
            
        except Exception as e:
            print(f"❌ Failed to fetch workflow {workflow_file}: {e}")
            return None
    
    def get_learning_summary(self) -> Dict:
        """Get summary of learning progress"""
        total_workflows = len(self.workflows)
        total_optimizations = sum(len(w.get('optimizations', [])) for w in self.workflows.values())
        
        return {
            'workflows_learned': total_workflows,
            'optimizations_identified': total_optimizations,
            'patterns_recognized': len(self.step_patterns),
            'learning_confidence': min(100, (total_workflows * 10) + (total_optimizations * 2))
        }