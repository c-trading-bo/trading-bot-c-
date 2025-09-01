#!/usr/bin/env python3
"""
🎯 Trading Bot Monitoring Dashboard
Complete monitoring solution for cloud learning and local trading.
"""

import json
import requests
import datetime
import argparse
import sys
from pathlib import Path

def check_github_actions(repo: str, token: str) -> dict:
    """Check GitHub Actions workflow status"""
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': f'token {token}'
    }
    
    # Get workflow runs
    url = f'https://api.github.com/repos/{repo}/actions/runs'
    response = requests.get(url, headers=headers, params={'per_page': 20})
    
    if response.status_code != 200:
        return {'error': f'GitHub API error: {response.status_code}'}
    
    runs_data = response.json()
    runs = runs_data.get('workflow_runs', [])
    
    # Filter training runs
    training_runs = [
        run for run in runs 
        if 'train' in run['name'].lower() or 'ml' in run['name'].lower()
    ]
    
    if not training_runs:
        return {'error': 'No training workflows found'}
    
    # Calculate metrics
    total_runs = len(training_runs)
    successful_runs = len([r for r in training_runs if r['conclusion'] == 'success'])
    failed_runs = len([r for r in training_runs if r['conclusion'] == 'failure'])
    
    success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
    
    # Get latest run info
    latest_run = training_runs[0] if training_runs else None
    
    return {
        'success_rate': success_rate,
        'total_runs': total_runs,
        'successful_runs': successful_runs,
        'failed_runs': failed_runs,
        'latest_run': {
            'status': latest_run['status'] if latest_run else 'unknown',
            'conclusion': latest_run['conclusion'] if latest_run else 'unknown',
            'created_at': latest_run['created_at'] if latest_run else None,
            'run_number': latest_run['run_number'] if latest_run else 0
        } if latest_run else None,
        'recent_runs': training_runs[:5]
    }

def check_local_bot() -> dict:
    """Check local bot status"""
    try:
        # Try to connect to local dashboard
        response = requests.get('http://localhost:5050/healthz', timeout=5)
        if response.status_code == 200:
            return {
                'status': 'running',
                'dashboard_url': 'http://localhost:5050/dashboard'
            }
        else:
            return {
                'status': 'unhealthy',
                'error': f'Health check failed: {response.status_code}'
            }
    except requests.exceptions.RequestException as e:
        return {
            'status': 'offline',
            'error': str(e)
        }

def check_model_freshness(cdn_url: str) -> dict:
    """Check model freshness from CDN"""
    try:
        manifest_url = f"{cdn_url}/models/current.json"
        response = requests.get(manifest_url, timeout=10)
        
        if response.status_code != 200:
            return {'error': f'Manifest not accessible: {response.status_code}'}
        
        manifest = response.json()
        timestamp = manifest.get('timestamp')
        
        if timestamp:
            model_time = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            age_minutes = (datetime.datetime.now(datetime.timezone.utc) - model_time).total_seconds() / 60
            
            if age_minutes < 30:
                freshness = 'fresh'
            elif age_minutes < 120:
                freshness = 'acceptable'
            else:
                freshness = 'stale'
            
            return {
                'freshness': freshness,
                'age_minutes': age_minutes,
                'version': manifest.get('version', 'unknown'),
                'training_samples': manifest.get('training_samples', 0)
            }
        else:
            return {'error': 'No timestamp in manifest'}
            
    except Exception as e:
        return {'error': f'Model check failed: {str(e)}'}

def generate_status_report(repo: str, token: str, cdn_url: str = None) -> dict:
    """Generate comprehensive status report"""
    report = {
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'github_actions': check_github_actions(repo, token),
        'local_bot': check_local_bot()
    }
    
    if cdn_url:
        report['models'] = check_model_freshness(cdn_url)
    
    # Overall health assessment
    github_ok = 'error' not in report['github_actions']
    local_ok = report['local_bot']['status'] == 'running'
    models_ok = cdn_url is None or 'error' not in report.get('models', {})
    
    if github_ok and local_ok and models_ok:
        overall_status = 'healthy'
    elif github_ok or local_ok:
        overall_status = 'degraded'
    else:
        overall_status = 'critical'
    
    report['overall_status'] = overall_status
    
    return report

def print_status_summary(report: dict):
    """Print a human-readable status summary"""
    print("🤖 Trading Bot Status Report")
    print("=" * 40)
    print(f"Generated: {report['timestamp']}")
    print(f"Overall Status: {report['overall_status'].upper()}")
    print()
    
    # GitHub Actions
    github = report['github_actions']
    if 'error' in github:
        print(f"❌ GitHub Actions: {github['error']}")
    else:
        print(f"☁️ GitHub Actions:")
        print(f"   Success Rate: {github['success_rate']:.1f}%")
        print(f"   Total Runs: {github['total_runs']}")
        print(f"   Latest: {github['latest_run']['conclusion'] if github['latest_run'] else 'none'}")
    
    print()
    
    # Local Bot
    local = report['local_bot']
    print(f"💻 Local Bot: {local['status'].upper()}")
    if local['status'] == 'running':
        print(f"   Dashboard: {local['dashboard_url']}")
    elif 'error' in local:
        print(f"   Error: {local['error']}")
    
    print()
    
    # Models
    if 'models' in report:
        models = report['models']
        if 'error' in models:
            print(f"📦 Models: {models['error']}")
        else:
            print(f"📦 Models:")
            print(f"   Freshness: {models['freshness'].upper()}")
            print(f"   Age: {models['age_minutes']:.1f} minutes")
            print(f"   Version: {models['version']}")
    
    print()

def main():
    parser = argparse.ArgumentParser(description='Trading Bot Monitoring Dashboard')
    parser.add_argument('--repo', required=True, help='GitHub repository (owner/name)')
    parser.add_argument('--token', required=True, help='GitHub token')
    parser.add_argument('--cdn-url', help='CDN base URL for model checking')
    parser.add_argument('--output', help='Output JSON file')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')
    
    args = parser.parse_args()
    
    try:
        report = generate_status_report(args.repo, args.token, args.cdn_url)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            if not args.quiet:
                print(f"✅ Report saved to {args.output}")
        
        if not args.quiet:
            print_status_summary(report)
        
        # Exit with error code if critical issues
        if report['overall_status'] == 'critical':
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()