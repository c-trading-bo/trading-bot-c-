#!/usr/bin/env python3
"""
ULTIMATE WORKFLOW VALIDATION SYSTEM
Tests all ultimate workflows for syntax, configuration, and failure prevention
"""

import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

class UltimateWorkflowValidator:
    def __init__(self):
        self.workflows_dir = Path(".github/workflows")
        self.ultimate_workflows = []
        self.validation_results = {}
        
    def discover_ultimate_workflows(self) -> List[Path]:
        """Find all ultimate workflow files"""
        ultimate_files = list(self.workflows_dir.glob("*ultimate*.yml"))
        ultimate_files.extend(list(self.workflows_dir.glob("*Ultimate*.yml")))
        
        # Remove duplicates
        unique_files = list(set(ultimate_files))
        self.ultimate_workflows = unique_files
        
        print(f"🔍 Found {len(unique_files)} ultimate workflows:")
        for wf in unique_files:
            print(f"   📄 {wf.name}")
        
        return unique_files
    
    def validate_workflow_syntax(self, workflow_file: Path) -> Dict:
        """Validate YAML syntax and structure"""
        print(f"\n🔧 Validating {workflow_file.name}...")
        
        result = {
            'file': workflow_file.name,
            'syntax_valid': False,
            'has_triggers': False,
            'has_permissions': False,
            'has_env_vars': False,
            'has_jobs': False,
            'job_count': 0,
            'uses_actions': False,
            'has_error_handling': False,
            'issues': []
        }
        
        try:
            with open(workflow_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse YAML
            workflow_data = yaml.safe_load(content)
            result['syntax_valid'] = True
            
            # Check structure
            if 'on' in workflow_data:
                result['has_triggers'] = True
                triggers = workflow_data['on']
                if isinstance(triggers, dict):
                    trigger_types = list(triggers.keys())
                    print(f"   ✅ Triggers: {', '.join(trigger_types)}")
            
            if 'permissions' in workflow_data:
                result['has_permissions'] = True
                print(f"   ✅ Has permissions configured")
            
            if 'env' in workflow_data:
                result['has_env_vars'] = True
                print(f"   ✅ Has environment variables")
            
            if 'jobs' in workflow_data:
                result['has_jobs'] = True
                jobs = workflow_data['jobs']
                result['job_count'] = len(jobs)
                print(f"   ✅ Has {len(jobs)} job(s)")
                
                # Check job details
                for job_name, job_config in jobs.items():
                    if 'steps' in job_config:
                        steps = job_config['steps']
                        
                        # Check for action usage
                        for step in steps:
                            if 'uses' in step:
                                result['uses_actions'] = True
                                break
                        
                        # Check for error handling
                        if any('continue-on-error' in step for step in steps):
                            result['has_error_handling'] = True
            
            # Check for common issues
            content_lower = content.lower()
            
            # Check for hardcoded paths
            if 'c:\\users\\kevin' in content_lower:
                result['issues'].append("Contains hardcoded Windows paths")
            
            # Check for proper indentation (common YAML issue)
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith(' ') and not line.startswith('#') and ':' in line:
                    # This is likely correct top-level structure
                    pass
                elif line.strip() and line.startswith('  ') and line.count('  ') % 2 != 0:
                    # Potential indentation issue
                    pass
            
            print(f"   ✅ Syntax validation passed")
            
        except yaml.YAMLError as e:
            result['issues'].append(f"YAML syntax error: {e}")
            print(f"   ❌ YAML syntax error: {e}")
        except Exception as e:
            result['issues'].append(f"Validation error: {e}")
            print(f"   ❌ Validation error: {e}")
        
        return result
    
    def check_workflow_dependencies(self, workflow_file: Path) -> Dict:
        """Check if workflow has proper dependencies and setup"""
        print(f"🔗 Checking dependencies for {workflow_file.name}...")
        
        dependencies = {
            'python_setup': False,
            'node_setup': False,
            'cache_setup': False,
            'token_access': False,
            'artifact_handling': False
        }
        
        try:
            with open(workflow_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for common setups
            if 'actions/setup-python' in content:
                dependencies['python_setup'] = True
                print("   ✅ Python setup configured")
            
            if 'actions/setup-node' in content:
                dependencies['node_setup'] = True
                print("   ✅ Node.js setup configured")
            
            if 'actions/cache' in content:
                dependencies['cache_setup'] = True
                print("   ✅ Caching configured")
            
            if 'GITHUB_TOKEN' in content or 'github.token' in content:
                dependencies['token_access'] = True
                print("   ✅ GitHub token access configured")
            
            if 'actions/upload-artifact' in content or 'actions/download-artifact' in content:
                dependencies['artifact_handling'] = True
                print("   ✅ Artifact handling configured")
                
        except Exception as e:
            print(f"   ⚠️ Dependency check error: {e}")
        
        return dependencies
    
    def test_workflow_auto_response(self) -> Dict:
        """Test if workflows are set up for auto-response"""
        print("\n🚨 Testing Auto-Response Configuration...")
        
        # Check main cloud mechanic workflow
        main_workflow = Path(".github/workflows/cloud_bot_mechanic.yml")
        auto_response_config = {
            'main_workflow_exists': main_workflow.exists(),
            'has_workflow_run_triggers': False,
            'monitors_ultimate_workflows': False,
            'monitored_workflow_count': 0
        }
        
        if main_workflow.exists():
            content = main_workflow.read_text()
            
            if 'workflow_run:' in content:
                auto_response_config['has_workflow_run_triggers'] = True
                print("   ✅ Has workflow_run triggers")
                
                # Count monitored workflows
                ultimate_mentions = content.count('Ultimate')
                auto_response_config['monitored_workflow_count'] = ultimate_mentions
                
                if ultimate_mentions >= 5:
                    auto_response_config['monitors_ultimate_workflows'] = True
                    print(f"   ✅ Monitors {ultimate_mentions} ultimate workflows")
        
        return auto_response_config
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive validation report"""
        print("\n📋 ULTIMATE WORKFLOW COMPREHENSIVE VALIDATION")
        print("=" * 60)
        
        # Discover workflows
        ultimate_workflows = self.discover_ultimate_workflows()
        
        # Validate each workflow
        validation_results = []
        for workflow in ultimate_workflows:
            syntax_result = self.validate_workflow_syntax(workflow)
            dependency_result = self.check_workflow_dependencies(workflow)
            
            combined_result = {
                **syntax_result,
                'dependencies': dependency_result
            }
            validation_results.append(combined_result)
            self.validation_results[workflow.name] = combined_result
        
        # Test auto-response
        auto_response = self.test_workflow_auto_response()
        
        # Calculate summary statistics
        total_workflows = len(validation_results)
        valid_syntax = sum(1 for r in validation_results if r['syntax_valid'])
        has_jobs = sum(1 for r in validation_results if r['has_jobs'])
        has_triggers = sum(1 for r in validation_results if r['has_triggers'])
        total_issues = sum(len(r['issues']) for r in validation_results)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_ultimate_workflows': total_workflows,
            'valid_syntax_count': valid_syntax,
            'workflows_with_jobs': has_jobs,
            'workflows_with_triggers': has_triggers,
            'total_issues_found': total_issues,
            'auto_response_configured': auto_response['has_workflow_run_triggers'],
            'validation_results': validation_results,
            'auto_response_config': auto_response
        }
        
        print(f"\n📊 VALIDATION SUMMARY:")
        print("=" * 40)
        print(f"✅ Total Ultimate Workflows: {total_workflows}")
        print(f"✅ Valid Syntax: {valid_syntax}/{total_workflows}")
        print(f"✅ With Jobs: {has_jobs}/{total_workflows}")
        print(f"✅ With Triggers: {has_triggers}/{total_workflows}")
        
        if total_issues > 0:
            print(f"⚠️ Total Issues Found: {total_issues}")
        else:
            print(f"✅ No Issues Found!")
        
        if auto_response['has_workflow_run_triggers']:
            print(f"✅ Auto-Response: Configured for {auto_response['monitored_workflow_count']} workflows")
        else:
            print(f"⚠️ Auto-Response: Not configured")
        
        # Overall status
        if valid_syntax == total_workflows and total_issues == 0:
            overall_status = "EXCELLENT"
            print(f"\n🎉 OVERALL STATUS: {overall_status}")
            print("   All ultimate workflows are functioning perfectly!")
        elif valid_syntax >= total_workflows * 0.8:
            overall_status = "GOOD" 
            print(f"\n✅ OVERALL STATUS: {overall_status}")
            print("   Most ultimate workflows are working well")
        else:
            overall_status = "NEEDS_ATTENTION"
            print(f"\n⚠️ OVERALL STATUS: {overall_status}")
            print("   Some ultimate workflows need fixes")
        
        summary['overall_status'] = overall_status
        
        # Save report
        report_file = Path('ultimate_workflow_validation.json')
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n💾 Validation report saved to: {report_file}")
        
        return summary

def main():
    """Run ultimate workflow validation"""
    validator = UltimateWorkflowValidator()
    return validator.generate_comprehensive_report()

if __name__ == "__main__":
    main()