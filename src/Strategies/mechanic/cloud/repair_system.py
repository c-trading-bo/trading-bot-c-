#!/usr/bin/env python3
"""
REPAIR SYSTEM - Automatic repair of broken workflows
"""

import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List

class RepairSystem:
    def __init__(self):
        self.cloud_path = Path('Intelligence/mechanic/cloud')
        self.workflow_path = Path('.github/workflows')
        self.repairs_made = []
        
    def repair_all_workflows(self):
        """Repair all broken workflows"""
        print("\n🔧 REPAIRING WORKFLOWS...")
        
        repair_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'repairs_attempted': [],
            'repairs_successful': [],
            'repairs_failed': [],
            'files_created': [],
            'files_modified': []
        }
        
        # Load workflow analysis to find broken workflows
        analysis_path = self.cloud_path / 'database' / 'workflows.json'
        analysis = self.load_json(analysis_path, {})
        
        # Repair broken workflows
        for wf_name, wf_data in analysis.items():
            if isinstance(wf_data, dict) and wf_data.get('issues'):
                repair_results['repairs_attempted'].append(wf_name)
                
                if self.repair_workflow(wf_name, wf_data):
                    repair_results['repairs_successful'].append(wf_name)
                    repair_results['files_modified'].append(wf_name)
                else:
                    repair_results['repairs_failed'].append(wf_name)
        
        # Create missing critical workflows
        missing_workflows = self.check_missing_critical_workflows()
        for missing_wf in missing_workflows:
            if self.create_missing_workflow(missing_wf):
                repair_results['files_created'].append(missing_wf)
                repair_results['repairs_successful'].append(missing_wf)
        
        repair_results['repairs_made'] = self.repairs_made
        
        # Save results
        output_path = self.cloud_path / 'database' / 'repairs.json'
        self.save_json(output_path, repair_results)
        
        self.print_repair_summary(repair_results)
        
        # Set GitHub output for changes
        if repair_results['files_created'] or repair_results['files_modified']:
            self.set_github_output('changes_made', 'true')
        
        return repair_results
    
    def repair_workflow(self, wf_name: str, wf_data: Dict) -> bool:
        """Repair a single workflow"""
        try:
            wf_path = self.workflow_path / wf_name
            
            # Load existing workflow or create new one
            if wf_path.exists():
                # First, check for YAML syntax issues by reading raw content
                with open(wf_path, 'r') as f:
                    raw_content = f.read()
                
                # Fix common YAML syntax errors
                if 'true:' in raw_content and 'on:' not in raw_content:
                    raw_content = raw_content.replace('true:', 'on:')
                    with open(wf_path, 'w') as f:
                        f.write(raw_content)
                    self.repairs_made.append(f"{wf_name}: Fixed YAML syntax (true: -> on:)")
                    print(f"    🔧 Fixed YAML syntax in {wf_name}")
                
                # Now load the corrected YAML
                with open(wf_path, 'r') as f:
                    workflow = yaml.safe_load(f) or {}
            else:
                workflow = {}
            
            modified = False
            
            # Fix workflow structure issues
            if True in workflow:  # Handle remaining True key issues
                workflow['on'] = workflow.pop(True)
                modified = True
                self.repairs_made.append(f"{wf_name}: Fixed workflow trigger structure")
            
            # Fix common issues
            for issue in wf_data.get('issues', []):
                if 'No jobs defined' in issue:
                    workflow['jobs'] = self.get_default_job()
                    modified = True
                    self.repairs_made.append(f"{wf_name}: Added default job")
                
                elif 'Empty workflow file' in issue:
                    workflow = self.get_basic_workflow_template(wf_name)
                    modified = True
                    self.repairs_made.append(f"{wf_name}: Created basic workflow structure")
                
                elif 'Invalid cron' in issue:
                    if 'on' not in workflow:
                        workflow['on'] = {}
                    workflow['on']['schedule'] = [{'cron': '0 */6 * * *'}]
                    modified = True
                    self.repairs_made.append(f"{wf_name}: Fixed cron expression")
            
            if modified:
                # Ensure basic structure
                if 'name' not in workflow:
                    workflow['name'] = wf_name.replace('.yml', '').replace('_', ' ').title()
                
                # Save repaired workflow
                with open(wf_path, 'w') as f:
                    yaml.dump(workflow, f, default_flow_style=False, sort_keys=False)
                
                print(f"    ✅ Repaired {wf_name}")
                return True
            
            return False
            
        except Exception as e:
            print(f"    ❌ Failed to repair {wf_name}: {e}")
            return False
    
    def check_missing_critical_workflows(self) -> List[str]:
        """Check for missing critical workflows"""
        critical_workflows = [
            'cloud_bot_mechanic.yml',
            'ci.yml',
            'quality-assurance.yml'
        ]
        
        missing = []
        for wf in critical_workflows:
            if not (self.workflow_path / wf).exists():
                missing.append(wf)
        
        return missing
    
    def create_missing_workflow(self, wf_name: str) -> bool:
        """Create a missing critical workflow"""
        try:
            template = self.get_workflow_template(wf_name)
            
            if template:
                wf_path = self.workflow_path / wf_name
                wf_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(wf_path, 'w') as f:
                    f.write(template)
                
                print(f"    ✅ Created missing workflow: {wf_name}")
                self.repairs_made.append(f"Created missing workflow: {wf_name}")
                return True
            
            return False
            
        except Exception as e:
            print(f"    ❌ Failed to create {wf_name}: {e}")
            return False
    
    def get_default_job(self) -> Dict:
        """Get a default job configuration"""
        return {
            'default-job': {
                'runs-on': 'ubuntu-latest',
                'timeout-minutes': 30,
                'steps': [
                    {'uses': 'actions/checkout@v4'},
                    {
                        'uses': 'actions/setup-python@v4',
                        'with': {
                            'python-version': '3.10',
                            'cache': 'pip'
                        }
                    },
                    {
                        'name': 'Install dependencies',
                        'run': 'pip install -r requirements.txt || echo "No requirements.txt found"'
                    },
                    {
                        'name': 'Run placeholder',
                        'run': 'echo "Workflow repaired - please customize this job"'
                    }
                ]
            }
        }
    
    def get_basic_workflow_template(self, wf_name: str) -> Dict:
        """Get a basic workflow template"""
        clean_name = wf_name.replace('.yml', '').replace('_', ' ').title()
        
        return {
            'name': clean_name,
            'on': {
                'workflow_dispatch': None,
                'schedule': [{'cron': '0 8 * * *'}]  # Daily at 8 AM UTC
            },
            'jobs': self.get_default_job()
        }
    
    def get_workflow_template(self, wf_name: str) -> str:
        """Get template for specific workflow types"""
        templates = {
            'cloud_bot_mechanic.yml': self.get_cloud_mechanic_template(),
            'ci.yml': self.get_ci_template(),
            'quality-assurance.yml': self.get_qa_template()
        }
        
        return templates.get(wf_name, '')
    
    def get_cloud_mechanic_template(self) -> str:
        """Get cloud bot mechanic workflow template"""
        return '''name: 🌩️ Cloud Bot Mechanic

on:
  schedule:
    - cron: '*/30 * * * *'  # Every 30 minutes
  workflow_dispatch:

jobs:
  mechanic:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          pip install pyyaml requests croniter pytz
      
      - name: Run Cloud Mechanic
        run: |
          python Intelligence/mechanic/cloud/cloud_mechanic_core.py
'''
    
    def get_ci_template(self) -> str:
        """Get CI workflow template"""
        return '''name: Continuous Integration

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt || echo "No requirements file"
      
      - name: Run tests
        run: |
          echo "Add your test commands here"
'''
    
    def get_qa_template(self) -> str:
        """Get quality assurance workflow template"""
        return '''name: Quality Assurance

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  quality-check:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          cache: 'pip'
      
      - name: Install QA tools
        run: |
          pip install flake8 black pylint
      
      - name: Run linting
        run: |
          echo "Add linting commands here"
          # flake8 . || true
          # black --check . || true
'''
    
    def print_repair_summary(self, results: Dict):
        """Print repair summary"""
        print("\n🔧 REPAIR SUMMARY")
        print(f"   Repairs attempted: {len(results['repairs_attempted'])}")
        print(f"   Repairs successful: {len(results['repairs_successful'])}")
        print(f"   Files created: {len(results['files_created'])}")
        print(f"   Files modified: {len(results['files_modified'])}")
        
        if results['repairs_failed']:
            print(f"   Repairs failed: {results['repairs_failed']}")
        
        if self.repairs_made:
            print("   Recent repairs:")
            for repair in self.repairs_made[-5:]:  # Show last 5
                print(f"     • {repair}")
    
    def set_github_output(self, name: str, value: str):
        """Set GitHub Actions output"""
        try:
            if os.environ.get('GITHUB_OUTPUT'):
                with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                    f.write(f"{name}={value}\n")
        except Exception as e:
            print(f"⚠️ Failed to set GitHub output {name}: {e}")
    
    def load_json(self, path: Path, default):
        """Load JSON with default fallback"""
        try:
            if path.exists():
                with open(path, 'r') as f:
                    return json.load(f)
        except:
            pass
        return default
    
    def save_json(self, path: Path, data: Dict):
        """Save JSON data"""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Failed to save {path}: {e}")

if __name__ == "__main__":
    repair_system = RepairSystem()
    repair_system.repair_all_workflows()
