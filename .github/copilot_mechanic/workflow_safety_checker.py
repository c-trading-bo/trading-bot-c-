#!/usr/bin/env python3
"""
WORKFLOW SAFETY CHECKER
Prevents AI from over-fixing working workflows
"""

import os
import yaml
from pathlib import Path

class WorkflowSafetyChecker:
    """
    Checks if a workflow should be modified before AI fixes it
    """
    
    def __init__(self):
        self.safe_patterns = [
            'name: Fixed Workflow',  # Already "fixed" - skip
            'echo "Workflow fixed"',  # Generic fix - skip  
            'uses: actions/checkout@v4',  # Standard action
        ]
        
        self.critical_workflows = [
            'copilot_ai_mechanic.yml',  # Don't modify the AI itself
            'build.yml',
            'dotnet.yml',
            'ci.yml'
        ]
    
    def should_modify_workflow(self, workflow_path: str) -> bool:
        """
        Check if this workflow should be modified
        """
        
        # Check if it's a critical workflow
        filename = Path(workflow_path).name
        if filename in self.critical_workflows:
            print(f"⚠️ SAFETY: Skipping critical workflow {filename}")
            return False
        
        # Read workflow content
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            return False
        
        # Check if already "fixed" with generic template
        if 'name: Fixed Workflow' in content and 'echo "Workflow fixed"' in content:
            print(f"⚠️ SAFETY: Workflow {filename} already has generic fix - skipping")
            return False
        
        # Check if it's actually working (has real steps)
        try:
            workflow_data = yaml.safe_load(content)
            
            # Count real steps vs generic ones
            total_steps = 0
            generic_steps = 0
            
            for job_name, job_data in workflow_data.get('jobs', {}).items():
                steps = job_data.get('steps', [])
                total_steps += len(steps)
                
                for step in steps:
                    if isinstance(step, dict):
                        run_cmd = step.get('run', '')
                        if 'echo "Workflow fixed"' in run_cmd:
                            generic_steps += 1
            
            # If more than 80% of steps are generic, skip
            if total_steps > 0 and (generic_steps / total_steps) > 0.8:
                print(f"⚠️ SAFETY: Workflow {filename} is mostly generic - skipping")
                return False
                
        except:
            pass
        
        print(f"✅ SAFETY: Workflow {filename} is safe to modify")
        return True
    
    def get_workflow_complexity(self, workflow_path: str) -> str:
        """
        Assess workflow complexity to determine confidence level
        """
        
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple complexity assessment
            if 'docker' in content.lower():
                return 'high'
            elif 'python' in content.lower() or 'node' in content.lower():
                return 'medium'
            else:
                return 'low'
                
        except:
            return 'unknown'

if __name__ == "__main__":
    checker = WorkflowSafetyChecker()
    
    # Test with some workflows
    workflow_dir = Path('.github/workflows')
    
    if workflow_dir.exists():
        for workflow_file in workflow_dir.glob('*.yml'):
            safe_to_modify = checker.should_modify_workflow(str(workflow_file))
            complexity = checker.get_workflow_complexity(str(workflow_file))
            
            print(f"{workflow_file.name}: Safe={safe_to_modify}, Complexity={complexity}")
    else:
        print("No .github/workflows directory found")
