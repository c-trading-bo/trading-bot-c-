#!/usr/bin/env python3
"""
🔧 WORKFLOW SCHEDULING DIAGNOSTIC & REPAIR TOOL
Analyzes all workflows to identify and fix scheduling issues
"""

import os
import yaml
import json
from pathlib import Path
from datetime import datetime

def check_workflow_scheduling():
    """
    Comprehensive workflow scheduling analysis
    """
    print("🔍 ANALYZING ALL WORKFLOW SCHEDULES...")
    print("=" * 60)
    
    workflow_dir = Path('.github/workflows')
    if not workflow_dir.exists():
        print("❌ .github/workflows directory not found!")
        return
    
    workflows = list(workflow_dir.glob('*.yml')) + list(workflow_dir.glob('*.yaml'))
    print(f"📊 Found {len(workflows)} workflow files")
    
    scheduled_workflows = []
    unscheduled_workflows = []
    broken_workflows = []
    
    for workflow_file in workflows:
        try:
            print(f"\n🔍 Analyzing: {workflow_file.name}")
            
            with open(workflow_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for BOM
            if content.startswith('\ufeff'):
                print("  ⚠️  UTF-8 BOM detected - needs fixing!")
                broken_workflows.append({
                    'file': workflow_file.name,
                    'issue': 'UTF-8 BOM encoding',
                    'severity': 'high'
                })
            
            # Parse YAML
            try:
                workflow_data = yaml.safe_load(content)
            except yaml.YAMLError as e:
                print(f"  ❌ YAML parse error: {e}")
                broken_workflows.append({
                    'file': workflow_file.name,
                    'issue': f'YAML syntax error: {e}',
                    'severity': 'critical'
                })
                continue
            
            # Check scheduling
            if 'on' in workflow_data:
                triggers = workflow_data['on']
                has_schedule = False
                
                if isinstance(triggers, dict):
                    if 'schedule' in triggers:
                        has_schedule = True
                        schedules = triggers['schedule']
                        if isinstance(schedules, list):
                            print(f"  ✅ Has {len(schedules)} scheduled triggers")
                            for i, sched in enumerate(schedules):
                                if 'cron' in sched:
                                    print(f"     Schedule {i+1}: {sched['cron']}")
                        else:
                            print(f"  ✅ Has 1 scheduled trigger: {schedules.get('cron', 'No cron found')}")
                    
                    # Check for other triggers
                    other_triggers = [k for k in triggers.keys() if k != 'schedule']
                    if other_triggers:
                        print(f"  📋 Other triggers: {', '.join(other_triggers)}")
                
                if has_schedule:
                    scheduled_workflows.append({
                        'file': workflow_file.name,
                        'name': workflow_data.get('name', 'Unnamed'),
                        'schedules': triggers.get('schedule', [])
                    })
                else:
                    unscheduled_workflows.append({
                        'file': workflow_file.name,
                        'name': workflow_data.get('name', 'Unnamed'),
                        'triggers': list(triggers.keys()) if isinstance(triggers, dict) else [str(triggers)]
                    })
            else:
                print("  ❌ No triggers defined!")
                broken_workflows.append({
                    'file': workflow_file.name,
                    'issue': 'No triggers defined',
                    'severity': 'critical'
                })
        
        except Exception as e:
            print(f"  ❌ Error processing {workflow_file.name}: {e}")
            broken_workflows.append({
                'file': workflow_file.name,
                'issue': f'Processing error: {e}',
                'severity': 'high'
            })
    
    # Generate comprehensive report
    print("\n" + "=" * 60)
    print("📊 WORKFLOW SCHEDULING ANALYSIS REPORT")
    print("=" * 60)
    
    print(f"\n✅ SCHEDULED WORKFLOWS ({len(scheduled_workflows)}):")
    for wf in scheduled_workflows:
        print(f"  • {wf['name']} ({wf['file']})")
    
    print(f"\n⚠️  UNSCHEDULED WORKFLOWS ({len(unscheduled_workflows)}):")
    for wf in unscheduled_workflows:
        triggers = ', '.join(wf['triggers'])
        print(f"  • {wf['name']} ({wf['file']}) - Triggers: {triggers}")
    
    print(f"\n❌ BROKEN WORKFLOWS ({len(broken_workflows)}):")
    for wf in broken_workflows:
        severity_icon = "🔥" if wf['severity'] == 'critical' else "⚠️"
        print(f"  {severity_icon} {wf['file']} - {wf['issue']}")
    
    # Calculate coverage
    total_workflows = len(workflows)
    scheduled_percentage = (len(scheduled_workflows) / total_workflows) * 100 if total_workflows > 0 else 0
    
    print(f"\n📈 SCHEDULING COVERAGE:")
    print(f"  Total workflows: {total_workflows}")
    print(f"  Scheduled: {len(scheduled_workflows)} ({scheduled_percentage:.1f}%)")
    print(f"  Unscheduled: {len(unscheduled_workflows)}")
    print(f"  Broken: {len(broken_workflows)}")
    
    # Save detailed report
    report = {
        'timestamp': datetime.utcnow().isoformat(),
        'total_workflows': total_workflows,
        'scheduled_count': len(scheduled_workflows),
        'unscheduled_count': len(unscheduled_workflows),
        'broken_count': len(broken_workflows),
        'scheduling_coverage_percentage': scheduled_percentage,
        'scheduled_workflows': scheduled_workflows,
        'unscheduled_workflows': unscheduled_workflows,
        'broken_workflows': broken_workflows
    }
    
    os.makedirs('Intelligence/data/mechanic', exist_ok=True)
    with open('Intelligence/data/mechanic/workflow_scheduling_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Report saved to: Intelligence/data/mechanic/workflow_scheduling_report.json")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    if broken_workflows:
        print("  1. Fix broken workflows immediately (critical for system reliability)")
    if unscheduled_workflows:
        print("  2. Add schedules to unscheduled workflows for 24/7 monitoring")
    if scheduled_percentage < 80:
        print("  3. Target: 80%+ workflows should have automated schedules")
    
    return report

def fix_utf8_bom_issues():
    """
    Fix UTF-8 BOM encoding issues in all workflow files
    """
    print("\n🔧 FIXING UTF-8 BOM ENCODING ISSUES...")
    print("=" * 40)
    
    workflow_dir = Path('.github/workflows')
    workflows = list(workflow_dir.glob('*.yml')) + list(workflow_dir.glob('*.yaml'))
    
    fixed_count = 0
    
    for workflow_file in workflows:
        try:
            with open(workflow_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if content.startswith('\ufeff'):
                print(f"🔧 Fixing BOM in: {workflow_file.name}")
                
                # Remove BOM and save
                clean_content = content[1:]  # Remove BOM character
                
                # Backup original
                backup_file = workflow_file.with_suffix(f'.bom_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yml')
                with open(backup_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                # Save clean version
                with open(workflow_file, 'w', encoding='utf-8') as f:
                    f.write(clean_content)
                
                fixed_count += 1
                print(f"  ✅ Fixed! Backup saved as: {backup_file.name}")
        
        except Exception as e:
            print(f"  ❌ Error fixing {workflow_file.name}: {e}")
    
    print(f"\n✅ Fixed UTF-8 BOM issues in {fixed_count} files")
    return fixed_count

if __name__ == "__main__":
    print("🚀 WORKFLOW SCHEDULING DIAGNOSTIC TOOL")
    print("=" * 60)
    
    # Run comprehensive analysis
    report = check_workflow_scheduling()
    
    # Fix BOM issues
    bom_fixes = fix_utf8_bom_issues()
    
    print(f"\n🎯 SUMMARY:")
    print(f"  • Analyzed {report['total_workflows']} workflows")
    print(f"  • {report['scheduled_count']} have schedules ({report['scheduling_coverage_percentage']:.1f}%)")
    print(f"  • {report['broken_count']} have critical issues")
    print(f"  • Fixed {bom_fixes} UTF-8 BOM issues")
    
    if report['broken_count'] > 0:
        print(f"\n⚠️  CRITICAL: {report['broken_count']} workflows need immediate attention!")
        print("   Your Ultimate Defense System may not be fully operational!")
    else:
        print(f"\n✅ All workflows are syntactically valid!")
        if report['scheduling_coverage_percentage'] >= 80:
            print("🎯 Excellent scheduling coverage - Defense System fully operational!")
        else:
            print("📈 Consider adding schedules to more workflows for better coverage")
