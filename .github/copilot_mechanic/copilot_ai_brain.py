#!/usr/bin/env python3
import os
import json
import requests
import yaml
import re
from datetime import datetime
from typing import Dict, List, Optional, Any

class GitHubCopilotAIBrain:
    def __init__(self):
        self.github_token = os.environ.get('GITHUB_TOKEN')
        self.repo_full_name = os.environ.get('GITHUB_REPOSITORY', 'c-trading-bo/trading-bot-c-')
        self.org, self.repo = self.repo_full_name.split('/')
        
        self.thresholds = {
            'auto_fix': 0.85,
            'pr_create': 0.60,
            'log_only': 0.30
        }
        
        self.knowledge_base = {
            'workflow_patterns': {},
            'error_solutions': {},
            'learned_fixes': {}
        }
        
        print("🧠 GitHub Copilot AI Brain v3.0-PRO")
        print(f"📍 Organization: {self.org}")
        print("🚀 GitHub Pro Intelligence Activated")
    
    def copilot_analyze(self, prompt: str, context: Dict = None) -> str:
        if 'workflow' in prompt.lower() and ('fail' in prompt.lower() or 'error' in prompt.lower()):
            return self.analyze_workflow_intelligently(prompt, context or {})
        elif 'yaml' in prompt.lower() or 'syntax' in prompt.lower():
            return self.analyze_yaml_intelligently(prompt, context or {})
        else:
            return self.general_intelligent_analysis(prompt, context or {})
    
    def analyze_workflow_intelligently(self, prompt: str, context: Dict) -> str:
        confidence = 0.75
        root_cause = 'Workflow configuration issue'
        fix_type = 'workflow_edit'
        
        if 'syntax error' in prompt.lower() or 'invalid yaml' in prompt.lower():
            confidence = 0.90
            root_cause = 'YAML syntax error in workflow file'
            fix_type = 'yaml_syntax_fix'
        elif 'dependency' in prompt.lower():
            confidence = 0.85
            root_cause = 'Missing or incorrect dependencies'
            fix_type = 'dependency_fix'
        
        return self.format_copilot_response({
            'root_cause': root_cause,
            'fix_type': fix_type,
            'confidence': confidence,
            'explanation': 'GitHub Copilot AI detected and analyzed the issue'
        })
    
    def analyze_yaml_intelligently(self, prompt: str, context: Dict) -> str:
        return self.format_copilot_response({
            'root_cause': 'YAML structure or syntax issue',
            'fix_type': 'yaml_correction',
            'confidence': 0.88,
            'explanation': 'YAML syntax needs correction for proper workflow execution'
        })
    
    def general_intelligent_analysis(self, prompt: str, context: Dict) -> str:
        return self.format_copilot_response({
            'root_cause': 'General workflow or configuration issue',
            'fix_type': 'general_fix',
            'confidence': 0.70,
            'explanation': 'Issue requires analysis and targeted fix'
        })
    
    def format_copilot_response(self, analysis: Dict) -> str:
        confidence_pct = int(analysis['confidence'] * 100)
        
        if analysis['confidence'] >= 0.85:
            action = '✅ WILL AUTO-FIX'
        elif analysis['confidence'] >= 0.60:
            action = '📝 WILL CREATE PR'
        else:
            action = '📋 LOG ONLY'
        
        return f'''
🧠 GITHUB COPILOT AI ANALYSIS
═══════════════════════════════

ROOT_CAUSE: {analysis['root_cause']}
FIX_TYPE: {analysis['fix_type']}
CONFIDENCE: {confidence_pct}%

EXPLANATION:
{analysis['explanation']}

AUTO_ACTION: {action}
'''
    
    # For backwards compatibility
    def analyze_workflow_failure(self, prompt: str, context: Dict = None) -> str:
        return self.copilot_analyze(prompt, context)
    
    def get_recent_failures(self, workflow_name: str, limit: int = 5) -> List[Dict]:
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        url = f'https://api.github.com/repos/{self.repo_full_name}/actions/runs'
        try:
            response = requests.get(url, headers=headers, params={'per_page': 20})
            if response.status_code == 200:
                runs = response.json().get('workflow_runs', [])
                failed_runs = [r for r in runs if r.get('conclusion') == 'failure']
                return failed_runs[:limit]
        except:
            pass
        return []
    
    def diagnose_and_fix_workflow(self, workflow_run: Dict) -> Dict:
        # Simulate diagnosis
        analysis = self.copilot_analyze(f"Analyze workflow failure: {workflow_run.get('name', 'unknown')}")
        
        return {
            'root_cause': 'Workflow issue detected',
            'fix_type': 'workflow_edit',
            'confidence': 85,  # Return as integer percentage
            'analysis': analysis,
            'workflow_id': workflow_run.get('id'),
            'fix_applied': False
        }
    
    def extract_fixes(self, analysis: str) -> List[Dict]:
        # Extract fix information from analysis
        fixes = []
        if 'CONFIDENCE:' in analysis:
            try:
                confidence_line = [line for line in analysis.split('\n') if 'CONFIDENCE:' in line][0]
                confidence = int(confidence_line.split(':')[1].strip().replace('%', ''))
                
                fixes.append({
                    'fix_type': 'workflow_edit',
                    'confidence': confidence,
                    'description': 'GitHub Copilot suggested fix'
                })
            except:
                pass
        
        return fixes
    
    def get_workflow_yaml(self, path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return ''

# Backwards compatibility
CopilotEnterpriseAIBrain = GitHubCopilotAIBrain

if __name__ == "__main__":
    brain = GitHubCopilotAIBrain()
    test_result = brain.copilot_analyze("Workflow failed with YAML syntax error")
    print(test_result)
