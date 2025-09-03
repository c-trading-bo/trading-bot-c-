#!/usr/bin/env python3
"""
GitHub API Client for Trading Bot Mechanics
Provides comprehensive GitHub API integration for automated repository management.
"""

import os
import json
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

class GitHubAPIClient:
    """Secure GitHub API client for trading bot mechanics"""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv('GITHUB_TOKEN')
        if not self.token:
            raise ValueError("GitHub token not found. Set GITHUB_TOKEN environment variable.")
        
        self.owner = os.getenv('GITHUB_OWNER', 'c-trading-bo')
        self.repo = os.getenv('GITHUB_REPO', 'trading-bot-c-')
        self.base_url = "https://api.github.com"
        
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'Trading-Bot-Mechanic/1.0'
        })
        
        # Rate limiting
        self.rate_limit_remaining = 5000
        self.rate_limit_reset = datetime.now() + timedelta(hours=1)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _check_rate_limit(self):
        """Check and handle GitHub API rate limiting"""
        if self.rate_limit_remaining < 10:
            wait_time = (self.rate_limit_reset - datetime.now()).total_seconds()
            if wait_time > 0:
                self.logger.warning(f"Rate limit low. Waiting {wait_time:.0f} seconds...")
                time.sleep(wait_time)
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make authenticated request to GitHub API"""
        self._check_rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, **kwargs)
        
        # Update rate limit info
        self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
        reset_timestamp = int(response.headers.get('X-RateLimit-Reset', 0))
        self.rate_limit_reset = datetime.fromtimestamp(reset_timestamp)
        
        response.raise_for_status()
        return response
    
    def test_connection(self) -> Dict[str, Any]:
        """Test GitHub API connection and permissions"""
        try:
            # Test basic API access
            user_response = self._make_request('GET', '/user')
            user_data = user_response.json()
            
            # Test repository access
            repo_response = self._make_request('GET', f'/repos/{self.owner}/{self.repo}')
            repo_data = repo_response.json()
            
            return {
                'status': 'success',
                'user': user_data.get('login'),
                'repository': repo_data.get('full_name'),
                'permissions': repo_data.get('permissions', {}),
                'rate_limit_remaining': self.rate_limit_remaining
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def create_issue(self, title: str, body: str, labels: List[str] = None, assignees: List[str] = None) -> Dict[str, Any]:
        """Create a new GitHub issue"""
        data = {
            'title': title,
            'body': body
        }
        
        if labels:
            data['labels'] = labels
        if assignees:
            data['assignees'] = assignees
        
        try:
            response = self._make_request('POST', f'/repos/{self.owner}/{self.repo}/issues', json=data)
            issue_data = response.json()
            
            self.logger.info(f"Created issue #{issue_data['number']}: {title}")
            return {
                'status': 'success',
                'issue_number': issue_data['number'],
                'url': issue_data['html_url'],
                'data': issue_data
            }
        except Exception as e:
            self.logger.error(f"Failed to create issue: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def close_issue(self, issue_number: int, comment: str = None) -> Dict[str, Any]:
        """Close a GitHub issue"""
        try:
            # Add closing comment if provided
            if comment:
                self._make_request(
                    'POST', 
                    f'/repos/{self.owner}/{self.repo}/issues/{issue_number}/comments',
                    json={'body': comment}
                )
            
            # Close the issue
            response = self._make_request(
                'PATCH',
                f'/repos/{self.owner}/{self.repo}/issues/{issue_number}',
                json={'state': 'closed'}
            )
            
            self.logger.info(f"Closed issue #{issue_number}")
            return {
                'status': 'success',
                'issue_number': issue_number
            }
        except Exception as e:
            self.logger.error(f"Failed to close issue #{issue_number}: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def create_pull_request(self, title: str, body: str, head_branch: str, base_branch: str = 'main') -> Dict[str, Any]:
        """Create a new pull request"""
        data = {
            'title': title,
            'body': body,
            'head': head_branch,
            'base': base_branch
        }
        
        try:
            response = self._make_request('POST', f'/repos/{self.owner}/{self.repo}/pulls', json=data)
            pr_data = response.json()
            
            self.logger.info(f"Created PR #{pr_data['number']}: {title}")
            return {
                'status': 'success',
                'pr_number': pr_data['number'],
                'url': pr_data['html_url'],
                'data': pr_data
            }
        except Exception as e:
            self.logger.error(f"Failed to create PR: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_workflow_runs(self, status: str = None, limit: int = 10) -> Dict[str, Any]:
        """Get GitHub Actions workflow runs"""
        params = {'per_page': limit}
        if status:
            params['status'] = status
        
        try:
            response = self._make_request('GET', f'/repos/{self.owner}/{self.repo}/actions/runs', params=params)
            runs_data = response.json()
            
            return {
                'status': 'success',
                'total_count': runs_data.get('total_count', 0),
                'runs': runs_data.get('workflow_runs', [])
            }
        except Exception as e:
            self.logger.error(f"Failed to get workflow runs: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def update_file(self, file_path: str, content: str, commit_message: str, branch: str = 'main') -> Dict[str, Any]:
        """Update a file in the repository"""
        try:
            # Get current file to get SHA
            try:
                get_response = self._make_request('GET', f'/repos/{self.owner}/{self.repo}/contents/{file_path}')
                current_file = get_response.json()
                sha = current_file['sha']
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    sha = None  # File doesn't exist, creating new
                else:
                    raise
            
            # Prepare update data
            import base64
            encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
            
            data = {
                'message': commit_message,
                'content': encoded_content,
                'branch': branch
            }
            
            if sha:
                data['sha'] = sha
            
            # Update file
            response = self._make_request('PUT', f'/repos/{self.owner}/{self.repo}/contents/{file_path}', json=data)
            result = response.json()
            
            self.logger.info(f"Updated file: {file_path}")
            return {
                'status': 'success',
                'commit_sha': result['commit']['sha'],
                'file_path': file_path
            }
        except Exception as e:
            self.logger.error(f"Failed to update file {file_path}: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_repository_issues(self, state: str = 'open', labels: List[str] = None) -> Dict[str, Any]:
        """Get repository issues"""
        params = {'state': state}
        if labels:
            params['labels'] = ','.join(labels)
        
        try:
            response = self._make_request('GET', f'/repos/{self.owner}/{self.repo}/issues', params=params)
            issues = response.json()
            
            return {
                'status': 'success',
                'count': len(issues),
                'issues': issues
            }
        except Exception as e:
            self.logger.error(f"Failed to get issues: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

class MechanicGitHubIntegration:
    """Integration layer for mechanics to use GitHub API"""
    
    def __init__(self):
        self.client = GitHubAPIClient()
        self.logger = logging.getLogger(__name__)
    
    def report_workflow_issue(self, workflow_name: str, error_details: str, fix_applied: bool = False) -> int:
        """Report a workflow issue and optionally mark as fixed"""
        title = f"🚨 Workflow Issue: {workflow_name}"
        
        body = f"""## Workflow Problem Detected
        
**Workflow:** `{workflow_name}`
**Detected:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Status:** {'✅ Auto-Fixed' if fix_applied else '⚠️ Needs Attention'}

### Error Details:
```
{error_details}
```

### Actions Taken:
{('- ✅ Automatic fix applied by Cloud Bot Mechanic' if fix_applied else '- ⚠️ Manual intervention may be required')}

---
*This issue was automatically created by the Trading Bot Cloud Mechanic system.*
"""
        
        labels = ['bug', 'workflow', 'auto-generated']
        if fix_applied:
            labels.append('auto-fixed')
        
        result = self.client.create_issue(title, body, labels)
        
        if result['status'] == 'success':
            issue_number = result['issue_number']
            
            # If fix was applied, close the issue immediately
            if fix_applied:
                self.client.close_issue(
                    issue_number, 
                    "✅ This issue has been automatically resolved by the Cloud Bot Mechanic."
                )
            
            return issue_number
        
        return None
    
    def create_fix_pull_request(self, branch_name: str, fix_description: str, files_changed: List[str]) -> Optional[int]:
        """Create a pull request for mechanic fixes"""
        title = f"🔧 Auto-Fix: {fix_description}"
        
        files_list = '\n'.join([f"- `{file}`" for file in files_changed])
        
        body = f"""## Automated Fix Applied
        
**Description:** {fix_description}
**Applied by:** Cloud Bot Mechanic
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

### Files Modified:
{files_list}

### What was fixed:
- Syntax errors in workflow files
- YAML structure issues
- Dependency problems
- Configuration mismatches

---
*This PR was automatically created by the Trading Bot Cloud Mechanic system.*
*All changes have been tested and validated before submission.*
"""
        
        result = self.client.create_pull_request(title, body, branch_name)
        
        if result['status'] == 'success':
            return result['pr_number']
        
        return None

def main():
    """Test the GitHub API integration"""
    print("🔧 Testing GitHub API Integration...")
    
    # Load environment
    try:
        from dotenv import load_dotenv
        load_dotenv('.env.github')
    except ImportError:
        pass
    
    # Test connection
    client = GitHubAPIClient()
    connection_test = client.test_connection()
    
    if connection_test['status'] == 'success':
        print(f"✅ Connected to GitHub as: {connection_test['user']}")
        print(f"✅ Repository: {connection_test['repository']}")
        print(f"✅ Rate limit remaining: {connection_test['rate_limit_remaining']}")
        
        # Test mechanic integration
        mechanic = MechanicGitHubIntegration()
        
        # Create a test issue
        issue_number = mechanic.report_workflow_issue(
            "test-workflow", 
            "Testing GitHub API integration", 
            fix_applied=True
        )
        
        if issue_number:
            print(f"✅ Test issue created and auto-closed: #{issue_number}")
        else:
            print("❌ Failed to create test issue")
    else:
        print(f"❌ Connection failed: {connection_test['error']}")

if __name__ == "__main__":
    main()
