#!/usr/bin/env python3
"""
CLOUD MECHANIC FEATURE AUDIT
Comprehensive check of all cloud mechanic features and capabilities
"""

import os
import json
from pathlib import Path
from datetime import datetime

def audit_cloud_mechanic_features():
    """Audit all cloud mechanic features and check if they're working"""
    
    print("🔍 CLOUD MECHANIC FEATURE AUDIT")
    print("=" * 60)
    print(f"Audit Time: {datetime.now().isoformat()}")
    print("")
    
    audit_results = {
        'timestamp': datetime.now().isoformat(),
        'basic_features': {},
        'ultimate_features': {},
        'files_status': {},
        'workflow_status': {},
        'recommendations': []
    }
    
    # Check basic cloud mechanic features
    print("📋 BASIC CLOUD MECHANIC FEATURES:")
    print("-" * 40)
    
    basic_features = {
        'workflow_analysis': 'Analyzes all workflow files for issues',
        'schedule_analysis': 'Checks cron schedules and calculates run frequency', 
        'minute_budget_tracking': 'Tracks GitHub Actions minute usage',
        'broken_workflow_detection': 'Identifies workflows with syntax errors',
        'performance_monitoring': 'Monitors workflow execution times',
        'auto_issue_detection': 'Finds common workflow problems',
        'yaml_validation': 'Validates workflow YAML syntax',
        'github_api_integration': 'Connects to GitHub API for live data',
        'report_generation': 'Generates JSON reports with findings',
        'alert_system': 'Creates alerts for critical issues'
    }
    
    for feature, description in basic_features.items():
        # Check if feature exists in code
        core_file = Path('Intelligence/mechanic/cloud/cloud_mechanic_core.py')
        if core_file.exists():
            # Try multiple encodings to handle special characters
            content = None
            for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
                try:
                    with open(core_file, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            if feature.replace('_', '') in content.replace('_', '').lower():
                audit_results['basic_features'][feature] = 'PRESENT'
                print(f"✅ {feature}: {description}")
            else:
                audit_results['basic_features'][feature] = 'MISSING'
                print(f"❌ {feature}: {description}")
        else:
            audit_results['basic_features'][feature] = 'FILE_MISSING'
            print(f"⚠️  {feature}: Core file missing")
    
    print("")
    print("🚀 ULTIMATE CLOUD MECHANIC FEATURES:")
    print("-" * 40)
    
    ultimate_features = {
        'workflow_learning': 'AI-powered workflow pattern recognition',
        'intelligent_optimization': 'Automatic workflow optimization suggestions',
        'dependency_caching': 'Pre-caches dependencies for faster builds',
        'failure_prediction': 'Predicts and prevents workflow failures',
        'performance_analysis': 'Advanced performance metrics and bottleneck detection',
        'auto_fixing': 'Automatically fixes common workflow issues',
        'knowledge_base': 'Learns from past failures and successes',
        'preemptive_preparation': 'Prepares workflows before execution',
        'critical_path_analysis': 'Identifies workflow dependencies and optimization opportunities',
        'ultimate_mode_activation': 'Enables advanced AI features when ULTIMATE_MODE=true'
    }
    
    for feature, description in ultimate_features.items():
        if core_file.exists() and content:
            # Check for ultimate features
            ultimate_indicators = [
                'CloudMechanicUltimate',
                'WorkflowLearner',
                'learn_all_workflows',
                'prepare_workflow_intelligent',
                'ULTIMATE_MODE'
            ]
            
            if any(indicator in content for indicator in ultimate_indicators):
                if feature.replace('_', '') in content.replace('_', '').lower():
                    audit_results['ultimate_features'][feature] = 'PRESENT'
                    print(f"✅ {feature}: {description}")
                else:
                    audit_results['ultimate_features'][feature] = 'PARTIAL'
                    print(f"⚠️  {feature}: {description}")
            else:
                audit_results['ultimate_features'][feature] = 'MISSING'
                print(f"❌ {feature}: {description}")
    
    print("")
    print("📁 FILE STATUS CHECK:")
    print("-" * 40)
    
    required_files = {
        '.github/workflows/cloud_bot_mechanic.yml': 'Main workflow file',
        'Intelligence/mechanic/cloud/cloud_mechanic_core.py': 'Core mechanic engine',
        'Intelligence/mechanic/cloud/workflow_learner.py': 'AI workflow learning system',
        'Intelligence/mechanic/ULTIMATE_CLOUD_MECHANIC_GUIDE.md': 'Documentation',
        'Intelligence/mechanic/cloud/reports/': 'Reports directory',
        'Intelligence/mechanic/cloud/database/': 'Database directory'
    }
    
    for file_path, description in required_files.items():
        path = Path(file_path)
        if path.exists():
            if path.is_file():
                size = path.stat().st_size
                audit_results['files_status'][file_path] = f'EXISTS ({size} bytes)'
                print(f"✅ {file_path}: {description} ({size} bytes)")
            else:
                audit_results['files_status'][file_path] = 'EXISTS (directory)'
                print(f"✅ {file_path}: {description} (directory)")
        else:
            audit_results['files_status'][file_path] = 'MISSING'
            print(f"❌ {file_path}: {description} - MISSING")
    
    print("")
    print("🔧 WORKFLOW CONFIGURATION CHECK:")
    print("-" * 40)
    
    workflow_file = Path('.github/workflows/cloud_bot_mechanic.yml')
    if workflow_file.exists():
        try:
            with open(workflow_file, 'r', encoding='utf-8') as f:
                workflow_content = f.read()
            
            checks = {
                'has_schedule': 'cron:' in workflow_content,
                'has_manual_trigger': 'workflow_dispatch:' in workflow_content,
                'calls_cloud_mechanic': 'cloud_mechanic_core.py' in workflow_content,
                'ultimate_mode_enabled': 'ULTIMATE_MODE' in workflow_content,
                'has_github_token': 'GITHUB_TOKEN' in workflow_content or 'github.token' in workflow_content,
                'has_permissions': 'permissions:' in workflow_content
            }
            
            for check, result in checks.items():
                audit_results['workflow_status'][check] = result
                status = "✅" if result else "❌"
                print(f"{status} {check}: {result}")
                
        except Exception as e:
            print(f"❌ Error reading workflow file: {e}")
            audit_results['workflow_status']['error'] = str(e)
    
    print("")
    print("💡 RECOMMENDATIONS:")
    print("-" * 40)
    
    recommendations = []
    
    # Check if ULTIMATE_MODE is enabled
    if not audit_results['workflow_status'].get('ultimate_mode_enabled', False):
        recommendations.append("Enable ULTIMATE_MODE in workflow to activate AI features")
        print("⚠️  ULTIMATE_MODE is not enabled - missing advanced AI features")
    
    # Check if workflow is calling the mechanic
    if not audit_results['workflow_status'].get('calls_cloud_mechanic', False):
        recommendations.append("Workflow should directly call cloud_mechanic_core.py")
        print("⚠️  Workflow not directly calling cloud_mechanic_core.py")
    
    # Check recent activity
    reports_dir = Path('Intelligence/mechanic/cloud/reports')
    if reports_dir.exists():
        latest_reports = list(reports_dir.glob('*.json'))
        if latest_reports:
            latest_report = max(latest_reports, key=lambda p: p.stat().st_mtime)
            age_days = (datetime.now().timestamp() - latest_report.stat().st_mtime) / 86400
            if age_days > 1:
                recommendations.append(f"Latest report is {age_days:.1f} days old - trigger manually")
                print(f"⚠️  Latest report is {age_days:.1f} days old")
        else:
            recommendations.append("No reports found - mechanic may not be running")
            print("⚠️  No reports found in reports directory")
    
    audit_results['recommendations'] = recommendations
    
    print("")
    print("📊 AUDIT SUMMARY:")
    print("-" * 40)
    
    basic_present = sum(1 for v in audit_results['basic_features'].values() if v == 'PRESENT')
    basic_total = len(audit_results['basic_features'])
    
    ultimate_present = sum(1 for v in audit_results['ultimate_features'].values() if v == 'PRESENT')
    ultimate_total = len(audit_results['ultimate_features'])
    
    files_present = sum(1 for v in audit_results['files_status'].values() if 'EXISTS' in v)
    files_total = len(audit_results['files_status'])
    
    workflow_working = sum(1 for v in audit_results['workflow_status'].values() if v is True)
    workflow_total = len([k for k in audit_results['workflow_status'].keys() if k != 'error'])
    
    print(f"✅ Basic Features: {basic_present}/{basic_total} ({basic_present/basic_total*100:.1f}%)")
    print(f"🚀 Ultimate Features: {ultimate_present}/{ultimate_total} ({ultimate_present/ultimate_total*100:.1f}%)")
    print(f"📁 Required Files: {files_present}/{files_total} ({files_present/files_total*100:.1f}%)")
    print(f"🔧 Workflow Config: {workflow_working}/{workflow_total} ({workflow_working/workflow_total*100:.1f}%)")
    
    # Save audit results
    audit_file = Path('cloud_mechanic_audit.json')
    with open(audit_file, 'w') as f:
        json.dump(audit_results, f, indent=2)
    
    print(f"\n💾 Audit results saved to: {audit_file}")
    
    return audit_results

if __name__ == "__main__":
    audit_cloud_mechanic_features()
