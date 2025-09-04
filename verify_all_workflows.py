#!/usr/bin/env python3
"""
Comprehensive verification script for all 27 workflows:
- Check for proper 24/7 scheduling
- Verify BotCore integration steps
- Analyze trading decision logic integration
"""

import os
import yaml
import json
from pathlib import Path
from datetime import datetime
import re

class WorkflowVerifier:
    def __init__(self, workflows_dir=".github/workflows"):
        self.workflows_dir = Path(workflows_dir)
        self.results = {
            "total_workflows": 0,
            "scheduled_workflows": 0,
            "integrated_workflows": 0,
            "missing_schedule": [],
            "missing_integration": [],
            "workflow_details": {},
            "verification_timestamp": datetime.utcnow().isoformat()
        }
    
    def analyze_schedule(self, workflow_content):
        """Analyze workflow scheduling configuration"""
        schedule_info = {
            "has_schedule": False,
            "has_cron": False,
            "cron_expressions": [],
            "session_coverage": {
                "asian": False,
                "european": False, 
                "us": False,
                "extended": False
            },
            "frequency_assessment": "none"
        }
        
        if 'on' in workflow_content:
            on_config = workflow_content['on']
            if 'schedule' in on_config:
                schedule_info["has_schedule"] = True
                schedule = on_config['schedule']
                
                if isinstance(schedule, list):
                    for item in schedule:
                        if 'cron' in item:
                            schedule_info["has_cron"] = True
                            cron_expr = item['cron']
                            schedule_info["cron_expressions"].append(cron_expr)
                            
                            # Analyze session coverage
                            self._analyze_session_coverage(cron_expr, schedule_info)
                
                # Assess frequency
                total_crons = len(schedule_info["cron_expressions"])
                if total_crons >= 10:
                    schedule_info["frequency_assessment"] = "high"
                elif total_crons >= 5:
                    schedule_info["frequency_assessment"] = "medium"
                elif total_crons >= 1:
                    schedule_info["frequency_assessment"] = "low"
        
        return schedule_info
    
    def _analyze_session_coverage(self, cron_expr, schedule_info):
        """Analyze which trading sessions are covered by cron expression"""
        # Extract hour patterns from cron (5th field)
        parts = cron_expr.split()
        if len(parts) >= 5:
            hour_part = parts[1]
            
            # Asian session: 18:00-23:59 CT (00:00-05:59 UTC)
            if any(h in hour_part for h in ['0', '1', '2', '3', '4', '5', '18', '19', '20', '21', '22', '23']):
                if any(h in hour_part for h in ['0', '1', '2', '3', '4', '5']):
                    schedule_info["session_coverage"]["asian"] = True
            
            # European session: 02:00-05:00 CT (08:00-11:00 UTC)  
            if any(h in hour_part for h in ['8', '9', '10', '11', '2', '3', '4', '5']):
                schedule_info["session_coverage"]["european"] = True
            
            # US session: 08:30-16:00 CT (14:30-22:00 UTC)
            if any(h in hour_part for h in ['14', '15', '16', '17', '18', '19', '20', '21', '22', '8', '9', '10', '11', '12', '13', '14', '15', '16']):
                schedule_info["session_coverage"]["us"] = True
            
            # Extended hours coverage
            if '*' in hour_part or '/' in hour_part:
                schedule_info["session_coverage"]["extended"] = True
    
    def analyze_integration(self, workflow_content):
        """Analyze BotCore integration configuration"""
        integration_info = {
            "has_integration": False,
            "integration_steps": [],
            "botcore_mentions": 0,
            "data_format_conversion": False,
            "git_commit_integration": False,
            "workflow_data_integration": False
        }
        
        workflow_str = yaml.dump(workflow_content).lower()
        
        # Check for BotCore mentions
        botcore_patterns = [
            'botcore', 'bot core', 'decision engine', 'trading decision',
            'workflow_data_integration', 'integrate with botcore'
        ]
        
        for pattern in botcore_patterns:
            integration_info["botcore_mentions"] += workflow_str.count(pattern.lower())
        
        # Check for integration steps
        if 'jobs' in workflow_content and workflow_content['jobs']:
            for job_name, job_config in workflow_content['jobs'].items():
                if job_config and 'steps' in job_config and job_config['steps']:
                    for step in job_config['steps']:
                        if isinstance(step, dict):
                            step_str = yaml.dump(step).lower()
                            
                            # Check for integration step patterns
                            if any(pattern in step_str for pattern in ['workflow_data_integration', 'integrate', 'botcore']):
                                integration_info["has_integration"] = True
                                integration_info["integration_steps"].append(step.get('name', 'Unnamed step'))
                                
                                if 'workflow_data_integration.py' in step_str:
                                    integration_info["workflow_data_integration"] = True
                                
                                if 'data format' in step_str or 'convert' in step_str:
                                    integration_info["data_format_conversion"] = True
                                
                                if 'git' in step_str and 'commit' in step_str:
                                    integration_info["git_commit_integration"] = True
        
        return integration_info
    
    def categorize_workflow(self, workflow_name, workflow_content):
        """Categorize workflow by purpose"""
        name_lower = workflow_name.lower()
        
        if any(term in name_lower for term in ['critical', 'trading', 'es_nq']):
            return 'critical_trading'
        elif any(term in name_lower for term in ['portfolio', 'heat', 'risk', 'positioning']):
            return 'risk_management'
        elif any(term in name_lower for term in ['news', 'sentiment', 'regime', 'intelligence', 'ml_rl']):
            return 'intelligence'
        elif any(term in name_lower for term in ['overnight', 'market_data', 'microstructure', 'volatility']):
            return 'market_analysis'
        elif any(term in name_lower for term in ['build', 'test', 'qa', 'ci', 'cloud']):
            return 'infrastructure'
        else:
            return 'other'
    
    def get_priority_level(self, category, schedule_info, integration_info):
        """Determine priority level for fixes"""
        if category in ['critical_trading', 'risk_management']:
            if not schedule_info["has_schedule"] or not integration_info["has_integration"]:
                return 'critical'
            elif schedule_info["frequency_assessment"] == 'low':
                return 'high'
        elif category == 'intelligence':
            if not schedule_info["has_schedule"] or not integration_info["has_integration"]:
                return 'high'
        elif category == 'market_analysis':
            if not schedule_info["has_schedule"]:
                return 'medium'
        
        return 'low'
    
    def verify_all_workflows(self):
        """Main verification function"""
        print("🔍 Starting comprehensive verification of all 27 workflows...")
        print("=" * 80)
        
        workflow_files = list(self.workflows_dir.glob("*.yml"))
        self.results["total_workflows"] = len(workflow_files)
        
        for workflow_file in sorted(workflow_files):
            workflow_name = workflow_file.stem
            
            try:
                # Try multiple encodings to handle various file formats
                for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                    try:
                        with open(workflow_file, 'r', encoding=encoding) as f:
                            workflow_content = yaml.safe_load(f)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise Exception("Could not decode file with any encoding")
                
                # Analyze scheduling
                schedule_info = self.analyze_schedule(workflow_content)
                if schedule_info["has_schedule"]:
                    self.results["scheduled_workflows"] += 1
                else:
                    self.results["missing_schedule"].append(workflow_name)
                
                # Analyze integration
                integration_info = self.analyze_integration(workflow_content)
                if integration_info["has_integration"]:
                    self.results["integrated_workflows"] += 1
                else:
                    self.results["missing_integration"].append(workflow_name)
                
                # Categorize and prioritize
                category = self.categorize_workflow(workflow_name, workflow_content)
                priority = self.get_priority_level(category, schedule_info, integration_info)
                
                # Store detailed results
                self.results["workflow_details"][workflow_name] = {
                    "category": category,
                    "priority": priority,
                    "schedule": schedule_info,
                    "integration": integration_info,
                    "file_path": str(workflow_file)
                }
                
                # Print status
                status_icon = "✅" if schedule_info["has_schedule"] and integration_info["has_integration"] else "❌"
                schedule_icon = "📅" if schedule_info["has_schedule"] else "⏰"
                integration_icon = "🔗" if integration_info["has_integration"] else "❌"
                
                print(f"{status_icon} {workflow_name:35} {schedule_icon} {integration_icon} [{category:15}] {priority:8}")
                
            except Exception as e:
                print(f"❌ Error processing {workflow_name}: {e}")
                continue
        
        self._print_summary()
        self._generate_action_plan()
        
        return self.results
    
    def _print_summary(self):
        """Print verification summary"""
        print("\n" + "=" * 80)
        print("📊 VERIFICATION SUMMARY")
        print("=" * 80)
        
        total = self.results["total_workflows"]
        scheduled = self.results["scheduled_workflows"]
        integrated = self.results["integrated_workflows"]
        
        print(f"📁 Total Workflows: {total}")
        print(f"📅 Scheduled: {scheduled}/{total} ({scheduled/total*100:.1f}%)")
        print(f"🔗 Integrated: {integrated}/{total} ({integrated/total*100:.1f}%)")
        print(f"⚠️  Missing Schedule: {len(self.results['missing_schedule'])}")
        print(f"⚠️  Missing Integration: {len(self.results['missing_integration'])}")
        
        if self.results["missing_schedule"]:
            print(f"\n❌ Workflows missing scheduling:")
            for name in self.results["missing_schedule"]:
                category = self.results["workflow_details"][name]["category"]
                priority = self.results["workflow_details"][name]["priority"]
                print(f"   • {name} [{category}] - Priority: {priority}")
        
        if self.results["missing_integration"]:
            print(f"\n❌ Workflows missing BotCore integration:")
            for name in self.results["missing_integration"]:
                category = self.results["workflow_details"][name]["category"]
                priority = self.results["workflow_details"][name]["priority"]
                print(f"   • {name} [{category}] - Priority: {priority}")
    
    def _generate_action_plan(self):
        """Generate prioritized action plan"""
        print("\n" + "=" * 80)
        print("🎯 ACTION PLAN")
        print("=" * 80)
        
        critical_issues = []
        high_issues = []
        medium_issues = []
        
        for name, details in self.results["workflow_details"].items():
            if details["priority"] == "critical":
                critical_issues.append((name, details))
            elif details["priority"] == "high":
                high_issues.append((name, details))
            elif details["priority"] == "medium":
                medium_issues.append((name, details))
        
        if critical_issues:
            print("🚨 CRITICAL PRIORITY (Fix immediately):")
            for name, details in critical_issues:
                missing = []
                if not details["schedule"]["has_schedule"]:
                    missing.append("scheduling")
                if not details["integration"]["has_integration"]:
                    missing.append("integration")
                print(f"   • {name} - Missing: {', '.join(missing)}")
        
        if high_issues:
            print("\n🔴 HIGH PRIORITY:")
            for name, details in high_issues:
                missing = []
                if not details["schedule"]["has_schedule"]:
                    missing.append("scheduling")
                if not details["integration"]["has_integration"]:
                    missing.append("integration")
                if details["schedule"]["frequency_assessment"] == "low":
                    missing.append("frequency optimization")
                print(f"   • {name} - Issues: {', '.join(missing) if missing else 'optimization needed'}")
        
        if medium_issues:
            print("\n🟡 MEDIUM PRIORITY:")
            for name, details in medium_issues:
                missing = []
                if not details["schedule"]["has_schedule"]:
                    missing.append("scheduling")
                if not details["integration"]["has_integration"]:
                    missing.append("integration")
                print(f"   • {name} - Issues: {', '.join(missing) if missing else 'minor improvements'}")
    
    def save_results(self, output_file="workflow_verification_results.json"):
        """Save detailed results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    verifier = WorkflowVerifier()
    results = verifier.verify_all_workflows()
    verifier.save_results()
    
    print("\n" + "=" * 80)
    print("✅ Verification complete! Check workflow_verification_results.json for details.")
    print("=" * 80)
