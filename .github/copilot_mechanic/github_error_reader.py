#!/usr/bin/env python3
"""
GITHUB WORKFLOW ERROR READER FOR AI BRAIN
Reads ACTUAL GitHub workflow errors for enhanced AI analysis
"""

import os
import json
import requests
import zipfile
import io
import re
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

class GitHubWorkflowErrorReader:
    """
    Reads ACTUAL GitHub workflow errors for AI analysis
    """
    
    def __init__(self):
        self.github_token = os.environ.get('GITHUB_TOKEN')
        self.repo = os.environ.get('GITHUB_REPOSITORY', 'c-trading-bo/trading-bot-c-')
        
        self.headers = {
            'Authorization': f'Bearer {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        print("🔍 GitHub Error Reader initialized")
    
    def get_failed_workflow_details(self, run_id: str) -> Dict:
        """
        Get COMPLETE error details from failed workflow
        """
        
        print(f"📋 Fetching details for workflow run {run_id}...")
        
        workflow_details = {
            'run_id': run_id,
            'error_messages': [],
            'failed_steps': [],
            'annotations': [],
            'logs': {},
            'conclusion': '',
            'github_error': ''
        }
        
        # 1. GET WORKFLOW RUN DETAILS
        run_url = f"https://api.github.com/repos/{self.repo}/actions/runs/{run_id}"
        
        run_response = requests.get(run_url, headers=self.headers)
        if run_response.status_code == 200:
            run_data = run_response.json()
            workflow_details['conclusion'] = run_data.get('conclusion')
            workflow_details['status'] = run_data.get('status')
            workflow_details['name'] = run_data.get('name')
        
        # 2. GET FAILED JOBS
        jobs_url = f"https://api.github.com/repos/{self.repo}/actions/runs/{run_id}/jobs"
        
        jobs_response = requests.get(jobs_url, headers=self.headers)
        if jobs_response.status_code == 200:
            jobs_data = jobs_response.json()
            
            for job in jobs_data.get('jobs', []):
                if job.get('conclusion') == 'failure':
                    
                    # Get failed steps
                    for step in job.get('steps', []):
                        if step.get('conclusion') == 'failure':
                            workflow_details['failed_steps'].append({
                                'name': step.get('name'),
                                'number': step.get('number'),
                                'status': step.get('status'),
                                'conclusion': step.get('conclusion')
                            })
        
        # 3. GET ACTUAL LOG CONTENT
        logs_url = f"https://api.github.com/repos/{self.repo}/actions/runs/{run_id}/logs"
        
        logs_response = requests.get(logs_url, headers=self.headers)
        
        if logs_response.status_code == 200:
            # GitHub returns logs as a zip file
            zip_content = io.BytesIO(logs_response.content)
            
            try:
                with zipfile.ZipFile(zip_content) as z:
                    for filename in z.namelist():
                        # Read each log file
                        with z.open(filename) as f:
                            log_content = f.read().decode('utf-8', errors='ignore')
                            
                            # Extract error lines
                            error_lines = self.extract_error_lines(log_content)
                            
                            if error_lines:
                                workflow_details['logs'][filename] = error_lines
                            
                            # Extract GitHub's specific error messages
                            github_errors = self.extract_github_errors(log_content)
                            workflow_details['error_messages'].extend(github_errors)
            
            except Exception as e:
                print(f"⚠️ Error parsing logs: {e}")
        
        # 4. GET WORKFLOW YAML TO UNDERSTAND WHAT FAILED
        workflow_path = self.get_workflow_path(run_id)
        if workflow_path:
            workflow_yaml = self.get_workflow_yaml(workflow_path)
            workflow_details['workflow_yaml'] = workflow_yaml
        
        return workflow_details
    
    def extract_error_lines(self, log_content: str) -> List[str]:
        """
        Extract error lines from log content
        """
        
        error_lines = []
        lines = log_content.splitlines()
        
        # Patterns that indicate errors
        error_patterns = [
            r'##\[error\]',  # GitHub Actions error annotation
            r'Error:',
            r'ERROR:',
            r'FAILED:',
            r'Fatal:',
            r'Traceback \(most recent call last\)',
            r'npm ERR!',
            r'yarn error',
            r'pip.*error',
            r'ModuleNotFoundError:',
            r'ImportError:',
            r'SyntaxError:',
            r'TypeError:',
            r'ValueError:',
            r'KeyError:',
            r'AttributeError:',
            r'IndentationError:',
            r'returned non-zero exit status',
            r'command not found',
            r'No such file or directory',
            r'Permission denied',
            r'timeout',
            r'connection refused'
        ]
        
        for i, line in enumerate(lines):
            for pattern in error_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Get context (5 lines before and after)
                    start = max(0, i - 5)
                    end = min(len(lines), i + 6)
                    context = lines[start:end]
                    error_lines.append({
                        'line_number': i,
                        'error_line': line,
                        'context': '\\n'.join(context)
                    })
                    break
        
        return error_lines
    
    def extract_github_errors(self, log_content: str) -> List[str]:
        """
        Extract GitHub Actions specific error messages
        """
        
        github_errors = []
        
        # GitHub Actions error format: ##[error]message
        error_matches = re.findall(r'##\[error\](.+)', log_content)
        github_errors.extend(error_matches)
        
        # GitHub Actions warning format: ##[warning]message
        warning_matches = re.findall(r'##\[warning\](.+)', log_content)
        github_errors.extend([f"Warning: {w}" for w in warning_matches])
        
        # Failed step indicators
        if "Process completed with exit code" in log_content:
            exit_code_matches = re.findall(r'Process completed with exit code (\d+)', log_content)
            for code in exit_code_matches:
                if code != '0':
                    github_errors.append(f"Process failed with exit code {code}")
        
        # Timeout errors
        if "The job was canceled because" in log_content:
            timeout_match = re.search(r'The job was canceled because (.+)', log_content)
            if timeout_match:
                github_errors.append(f"Job canceled: {timeout_match.group(1)}")
        
        return github_errors
    
    def get_workflow_path(self, run_id: str) -> Optional[str]:
        """
        Get the workflow file path
        """
        
        run_url = f"https://api.github.com/repos/{self.repo}/actions/runs/{run_id}"
        response = requests.get(run_url, headers=self.headers)
        
        if response.status_code == 200:
            run_data = response.json()
            return run_data.get('path')
        
        return None
    
    def get_workflow_yaml(self, workflow_path: str) -> str:
        """
        Get workflow YAML content
        """
        
        try:
            # Get from repository
            content_url = f"https://api.github.com/repos/{self.repo}/contents/{workflow_path}"
            response = requests.get(content_url, headers=self.headers)
            
            if response.status_code == 200:
                content_data = response.json()
                import base64
                yaml_content = base64.b64decode(content_data['content']).decode('utf-8')
                return yaml_content
        except:
            pass
        
        return ""
    
    def create_ai_diagnosis_prompt(self, workflow_details: Dict) -> str:
        """
        Create comprehensive prompt for AI with all error details
        """
        
        prompt = f"""
GITHUB WORKFLOW FAILURE ANALYSIS

Repository: {self.repo}
Run ID: {workflow_details['run_id']}
Workflow: {workflow_details.get('name', 'Unknown')}
Conclusion: {workflow_details.get('conclusion', 'failure')}

FAILED STEPS:
{json.dumps(workflow_details['failed_steps'], indent=2)}

GITHUB ERROR MESSAGES:
{chr(10).join(workflow_details['error_messages'][:10])}

ERROR LOG EXCERPTS:
"""
        
        # Add relevant log excerpts
        for filename, errors in list(workflow_details['logs'].items())[:3]:
            prompt += f"\\n\\nFrom {filename}:\\n"
            for error in errors[:2]:
                prompt += f"Line {error['line_number']}: {error['error_line']}\\n"
                prompt += f"Context:\\n{error['context'][:500]}\\n"
        
        # Add workflow YAML if available
        if workflow_details.get('workflow_yaml'):
            prompt += f"\\n\\nWORKFLOW YAML:\\n```yaml\\n{workflow_details['workflow_yaml'][:2000]}\\n```\\n"
        
        prompt += """

ANALYZE AND PROVIDE:
1. Root cause of the failure
2. Exact fix needed (with code)
3. Which file(s) to modify
4. Confidence level (0-100)

Format:
ROOT_CAUSE: <explanation>
FIX_TYPE: <workflow_yaml|code|dependency|config>
FILES_TO_FIX: <list of files>
FIX_CODE:
```yaml/python/json
<exact fix code>
```
CONFIDENCE: <0-100>
"""
        
        return prompt


# UPDATE THE MAIN AI BRAIN TO USE THIS READER
class EnhancedCopilotAIBrain:
    """
    Enhanced AI Brain that properly reads GitHub errors and auto-fixes
    """
    
    def __init__(self):
        self.error_reader = GitHubWorkflowErrorReader()
        self.repo = os.environ.get('GITHUB_REPOSITORY', 'c-trading-bo/trading-bot-c-')
        self.github_token = os.environ.get('GITHUB_TOKEN')
        self.headers = {
            'Authorization': f'Bearer {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        print("🧠 Enhanced AI Brain initialized with GitHub integration")
    
    def diagnose_workflow_failure_properly(self, run_id: str) -> Dict:
        """
        Properly diagnose workflow failure with ACTUAL GitHub errors
        """
        
        print(f"🧠 AI analyzing workflow run {run_id}...")
        
        # Get ACTUAL error details from GitHub
        workflow_details = self.error_reader.get_failed_workflow_details(run_id)
        
        # Create comprehensive prompt with real errors
        diagnosis_prompt = self.error_reader.create_ai_diagnosis_prompt(workflow_details)
        
        print("🔍 Analyzing error patterns...")
        
        # AI analysis (simplified for now, can integrate with OpenAI later)
        fix_data = self.analyze_errors_locally(workflow_details)
        
        # Apply fix if confidence is high
        if fix_data['confidence'] >= 85:
            print(f"✅ AI confidence: {fix_data['confidence']}%")
            print(f"📝 Root cause: {fix_data.get('root_cause', 'Unknown')}")
            print(f"🔧 Applying fix...")
            
            success = self.apply_targeted_fix(fix_data, workflow_details)
            
            if success:
                print("🎉 Fix applied successfully!")
                
                # Re-run workflow to test
                self.rerun_workflow(run_id)
            else:
                print("⚠️ Creating issue for manual review...")
                self.create_fix_issue(fix_data, workflow_details)
        else:
            print(f"⚠️ Low confidence ({fix_data['confidence']}%), creating issue for review")
            self.create_fix_issue(fix_data, workflow_details)
        
        return {
            'diagnosis': diagnosis_prompt,
            'fix_data': fix_data,
            'workflow_details': workflow_details
        }
    
    def analyze_errors_locally(self, workflow_details: Dict) -> Dict:
        """
        Local error analysis with pattern matching
        """
        
        confidence = 0
        root_cause = "Unknown error"
        fix_type = "manual"
        files_to_fix = []
        fix_code = ""
        
        error_messages = workflow_details.get('error_messages', [])
        failed_steps = workflow_details.get('failed_steps', [])
        annotations = workflow_details.get('annotations', [])
        
        # Common error patterns and fixes
        if any('permissions' in msg.lower() for msg in error_messages):
            confidence = 95
            root_cause = "Missing GitHub permissions"
            fix_type = "workflow_yaml"
            files_to_fix = [f".github/workflows/{workflow_details.get('name', 'unknown')}.yml"]
            fix_code = """
permissions:
  contents: write
  actions: write
  pull-requests: write
  issues: write
  id-token: write"""
        
        elif any('timeout' in msg.lower() or 'timed out' in msg.lower() for msg in error_messages):
            confidence = 90
            root_cause = "Workflow timeout"
            fix_type = "workflow_yaml" 
            fix_code = "timeout-minutes: 30"
        
        elif any('module not found' in msg.lower() or 'importerror' in msg.lower() for msg in error_messages):
            confidence = 85
            root_cause = "Missing Python dependencies"
            fix_type = "dependency"
            fix_code = "pip install missing-package"
        
        elif any('yaml' in msg.lower() and 'error' in msg.lower() for msg in error_messages):
            confidence = 80
            root_cause = "YAML syntax error"
            fix_type = "workflow_yaml"
        
        elif any('botcore' in msg.lower() for msg in error_messages):
            confidence = 88
            root_cause = "Missing BotCore integration"
            fix_type = "workflow_yaml"
            fix_code = """
      - name: "🔗 Integrate with BotCore Decision Engine"
        run: |
          echo "🔗 Converting workflow data to BotCore format..."
          python Intelligence/scripts/workflow_data_integration.py \\
            --workflow-type "workflow_name" \\
            --data-path "Intelligence/data/workflow_results.json" \\
            --output-path "Intelligence/data/integrated/workflow_name.json"
          echo "✅ BotCore integration complete\""""
        
        return {
            'confidence': confidence,
            'root_cause': root_cause,
            'fix_type': fix_type,
            'files_to_fix': files_to_fix,
            'fix_code': fix_code
        }
    
    def apply_targeted_fix(self, fix_data: Dict, workflow_details: Dict) -> bool:
        """
        Apply fix to the exact location of the error
        """
        
        try:
            files_to_fix = fix_data.get('files_to_fix', [])
            
            if not files_to_fix:
                # Determine from workflow details
                workflow_name = workflow_details.get('name', 'unknown')
                files_to_fix = [f".github/workflows/{workflow_name.lower().replace(' ', '_')}.yml"]
            
            for file_path in files_to_fix:
                if os.path.exists(file_path):
                    # Backup
                    backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        original_content = f.read()
                    
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.write(original_content)
                    
                    # Apply fix based on type
                    if fix_data.get('fix_type') == 'workflow_yaml':
                        fixed_content = self.apply_yaml_fix(original_content, fix_data)
                        
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(fixed_content)
                        
                        print(f"  ✅ Fixed {file_path}")
                        return True
            
            return False
        
        except Exception as e:
            print(f"❌ Error applying fix: {e}")
            return False
    
    def apply_yaml_fix(self, content: str, fix_data: Dict) -> str:
        """
        Apply YAML-specific fixes
        """
        
        lines = content.split('\n')
        
        # Add permissions if missing
        if 'permissions' in fix_data.get('fix_code', ''):
            if 'permissions:' not in content:
                # Find insertion point after 'on:' section
                for i, line in enumerate(lines):
                    if line.strip().startswith('on:'):
                        # Find end of 'on:' section
                        for j in range(i + 1, len(lines)):
                            if lines[j] and not lines[j].startswith(' ') and not lines[j].startswith('#'):
                                # Insert permissions here
                                permissions_lines = [
                                    '',
                                    'permissions:',
                                    '  contents: write',
                                    '  actions: write',
                                    '  pull-requests: write',
                                    '  issues: write',
                                    '  id-token: write'
                                ]
                                lines = lines[:j] + permissions_lines + lines[j:]
                                break
                        break
        
        # Add timeout if missing
        if 'timeout-minutes' in fix_data.get('fix_code', ''):
            for i, line in enumerate(lines):
                if 'runs-on:' in line:
                    if 'timeout-minutes:' not in content:
                        lines.insert(i + 1, '    timeout-minutes: 30')
                    break
        
        # Add BotCore integration
        if 'BotCore' in fix_data.get('fix_code', ''):
            # Find last step and add BotCore before commit
            for i in range(len(lines) - 1, -1, -1):
                if '- name: "💾 Commit Results"' in lines[i] or '- name: "Commit Results"' in lines[i]:
                    botcore_step = [
                        '',
                        '      - name: "🔗 Integrate with BotCore Decision Engine"',
                        '        run: |',
                        '          echo "🔗 Converting workflow data to BotCore format..."',
                        '          python Intelligence/scripts/workflow_data_integration.py \\',
                        '            --workflow-type "workflow_analysis" \\',
                        '            --data-path "Intelligence/data/workflow_results.json" \\',
                        '            --output-path "Intelligence/data/integrated/workflow_analysis.json"',
                        '          echo "✅ BotCore integration complete"'
                    ]
                    lines = lines[:i] + botcore_step + lines[i:]
                    break
        
        return '\n'.join(lines)
    
    def rerun_workflow(self, run_id: str):
        """
        Re-run the workflow after fix
        """
        
        try:
            rerun_url = f"https://api.github.com/repos/{self.repo}/actions/runs/{run_id}/rerun"
            
            response = requests.post(rerun_url, headers=self.headers)
            
            if response.status_code == 201:
                print("🔄 Workflow re-run triggered to test fix")
            else:
                print(f"⚠️ Could not re-run workflow: {response.status_code}")
        
        except Exception as e:
            print(f"❌ Error re-running workflow: {e}")
    
    def create_fix_issue(self, fix_data: Dict, workflow_details: Dict):
        """
        Create GitHub issue for manual review
        """
        
        try:
            issue_title = f"🚨 Workflow Failure Requires Manual Fix: {workflow_details.get('name', 'Unknown')}"
            
            issue_body = f"""
**Workflow Failure Analysis**

**Run ID:** {workflow_details.get('run_id')}
**Confidence:** {fix_data.get('confidence', 0)}%
**Root Cause:** {fix_data.get('root_cause', 'Unknown')}

**Error Details:**
{chr(10).join(workflow_details.get('error_messages', [])[:5])}

**Failed Steps:**
{json.dumps(workflow_details.get('failed_steps', []), indent=2)}

**Suggested Fix:**
```{fix_data.get('fix_type', 'yaml')}
{fix_data.get('fix_code', 'Manual investigation required')}
```

**Files to Check:**
{chr(10).join(f"- {f}" for f in fix_data.get('files_to_fix', []))}

*Generated by Ultimate AI+Cloud Bot Mechanic Defense System*
"""
            
            issue_url = f"https://api.github.com/repos/{self.repo}/issues"
            
            issue_data = {
                'title': issue_title,
                'body': issue_body,
                'labels': ['workflow-failure', 'ai-analysis', 'urgent']
            }
            
            response = requests.post(issue_url, headers=self.headers, json=issue_data)
            
            if response.status_code == 201:
                print("📋 Issue created for manual review")
            else:
                print(f"⚠️ Could not create issue: {response.status_code}")
        
        except Exception as e:
            print(f"❌ Error creating issue: {e}")
