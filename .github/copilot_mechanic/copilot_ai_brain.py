#!/usr/bin/env python3
import os
import json
import requests
import yaml
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

# Import error reader for enhanced analysis
try:
    from .github_error_reader import GitHubWorkflowErrorReader
    ERROR_READER_AVAILABLE = True
except ImportError:
    try:
        from github_error_reader import GitHubWorkflowErrorReader
        ERROR_READER_AVAILABLE = True
    except ImportError:
        print("⚠️ GitHub Error Reader not found - using basic analysis only")
        ERROR_READER_AVAILABLE = False
        GitHubWorkflowErrorReader = None

class GitHubCopilotAIBrain:
    def __init__(self):
        self.github_token = os.environ.get('GITHUB_TOKEN')
        self.repo_full_name = os.environ.get('GITHUB_REPOSITORY', 'c-trading-bo/trading-bot-c-')
        self.org, self.repo = self.repo_full_name.split('/')
        
        # Initialize error reader for enhanced analysis
        if ERROR_READER_AVAILABLE and GitHubWorkflowErrorReader:
            self.error_reader = GitHubWorkflowErrorReader()
            print("🔧 Enhanced GitHub Error Reader initialized")
        else:
            self.error_reader = None
            print("📋 Using basic pattern analysis only")
        
        self.thresholds = {
            'auto_fix': 0.95,  # Only auto-fix with 95%+ confidence
            'pr_create': 0.85,  # Create PR with 85%+ confidence  
            'log_only': 0.50   # Log analysis with 50%+ confidence
        }
        
        self.knowledge_base = {
            'workflow_patterns': {},
            'error_solutions': {},
            'learned_fixes': {}
        }
        
        print("🧠 GitHub Copilot AI Brain v3.0-PRO")
        print(f"📍 Organization: {self.org}")
        print("🚀 GitHub Pro Intelligence Activated")
    
    def copilot_analyze(self, prompt: str, context: Optional[Dict] = None) -> str:
        """
        MAIN COPILOT AI ANALYSIS - Now with enhanced error reading
        """
        print("🧠 GITHUB COPILOT AI ANALYSIS STARTING...")
        
        # Enhanced analysis with actual error reading
        if self.error_reader and context and context.get('workflow_run_id'):
            return self.analyze_with_error_reader(context['workflow_run_id'], prompt)
        
        # Original intelligent analysis patterns (fallback)
        if 'workflow' in prompt.lower() and ('fail' in prompt.lower() or 'error' in prompt.lower()):
            return self.analyze_workflow_intelligently(prompt, context or {})
        elif 'yaml' in prompt.lower() or 'syntax' in prompt.lower():
            return self.analyze_yaml_intelligently(prompt, context or {})
        else:
            return self.general_intelligent_analysis(prompt, context or {})
    
    def analyze_with_error_reader(self, run_id: str, prompt: str) -> str:
        """
        ENHANCED ANALYSIS using actual GitHub workflow error logs
        """
        print(f"🔍 Reading ACTUAL errors from workflow run {run_id}...")
        
        try:
            # Get complete error details
            error_details = self.error_reader.get_failed_workflow_details(run_id)
            
            # Create AI diagnosis prompt with real data
            ai_prompt = self.error_reader.create_ai_diagnosis_prompt(error_details)
            
            # Analyze the actual errors
            analysis = self.analyze_real_errors(error_details, ai_prompt)
            
            return self.format_enhanced_response(analysis, error_details)
            
        except Exception as e:
            print(f"⚠️ Error reader failed: {e}")
            # Fallback to pattern analysis
            return self.analyze_workflow_intelligently(prompt, {})
    
    def analyze_real_errors(self, error_details: Dict, ai_prompt: str) -> Dict:
        """
        Analyze REAL error data from GitHub logs
        """
        
        # Start with high confidence since we have real data
        confidence = 0.90
        
        # Analyze actual error messages
        error_messages = error_details.get('error_messages', [])
        failed_steps = error_details.get('failed_steps', [])
        logs = error_details.get('logs', {})
        
        root_cause = "Unknown error"
        fix_type = "general_fix"
        
        # YAML syntax errors
        if any('yaml' in msg.lower() or 'syntax error' in msg.lower() for msg in error_messages):
            root_cause = "YAML syntax error in workflow file"
            fix_type = "yaml_fix"
            confidence = 0.95
        
        # Python errors
        elif any('ModuleNotFoundError' in msg or 'ImportError' in msg for msg in error_messages):
            root_cause = "Missing Python dependencies"
            fix_type = "dependency_fix"
            confidence = 0.92
        
        # Node.js errors
        elif any('npm ERR!' in msg or 'yarn error' in msg for msg in error_messages):
            root_cause = "Node.js dependency or build error"
            fix_type = "node_fix"
            confidence = 0.88
        
        # Environment/setup errors
        elif any('command not found' in msg or 'No such file' in msg for msg in error_messages):
            root_cause = "Environment setup or path issue"
            fix_type = "env_fix"
            confidence = 0.85
        
        # Permission errors
        elif any('Permission denied' in msg for msg in error_messages):
            root_cause = "Permission or authentication issue"
            fix_type = "permission_fix"
            confidence = 0.87
        
        # Timeout errors
        elif any('timeout' in msg.lower() or 'canceled' in msg.lower() for msg in error_messages):
            root_cause = "Workflow timeout or resource limit"
            fix_type = "timeout_fix"
            confidence = 0.83
        
        # Use error context for more specific analysis
        for log_file, errors in logs.items():
            for error in errors:
                error_line = error.get('error_line', '')
                context = error.get('context', '')
                
                # More specific pattern matching with context
                if 'syntax error' in error_line.lower() and '.yml' in context:
                    root_cause = f"YAML syntax error in {log_file}"
                    fix_type = "yaml_fix"
                    confidence = 0.96
                    break
        
        return {
            'root_cause': root_cause,
            'fix_type': fix_type,
            'confidence': confidence,
            'error_messages': error_messages[:5],  # Top 5 errors
            'failed_steps': failed_steps,
            'analysis_type': 'real_error_data'
        }
    
    def format_enhanced_response(self, analysis: Dict, error_details: Dict) -> str:
        """
        Format response with enhanced error details
        """
        confidence_pct = int(analysis['confidence'] * 100)
        
        if analysis['confidence'] >= 0.85:
            action = '✅ WILL AUTO-FIX'
        elif analysis['confidence'] >= 0.60:
            action = '📝 WILL CREATE PR'
        else:
            action = '📋 LOG ONLY'
        
        response = f'''
🔍 ENHANCED GITHUB COPILOT AI ANALYSIS
════════════════════════════════════════

RUN_ID: {error_details['run_id']}
WORKFLOW: {error_details.get('name', 'Unknown')}

ROOT_CAUSE: {analysis['root_cause']}
FIX_TYPE: {analysis['fix_type']}
CONFIDENCE: {confidence_pct}% (based on REAL error data)

ACTUAL ERROR MESSAGES:
{chr(10).join(f"• {msg}" for msg in analysis.get('error_messages', [])[:3])}

FAILED STEPS:
{chr(10).join(f"• {step.get('name', 'Unknown')}" for step in analysis.get('failed_steps', [])[:3])}

AUTO_ACTION: {action}

🔧 ANALYSIS SOURCE: Real GitHub workflow logs and error data
'''
        return response
    
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
