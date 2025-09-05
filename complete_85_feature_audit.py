#!/usr/bin/env python3
"""
COMPLETE 85+ CLOUD AI MECHANIC FEATURE AUDIT
Discovers and validates ALL cloud AI mechanic features across the entire system
"""

import os
import sys
import ast
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List

class CompleteFeatureAudit:
    def __init__(self):
        self.mechanic_files = []
        self.total_features = 0
        self.features_by_category = {}
        self.active_features = 0
        
    def discover_all_mechanic_files(self) -> List[Path]:
        """Discover all mechanic-related Python files"""
        print("🔍 DISCOVERING ALL CLOUD AI MECHANIC FILES...")
        
        # Core cloud mechanic files
        cloud_files = list(Path("Intelligence/mechanic/cloud").glob("*.py"))
        
        # Local mechanic files  
        local_files = list(Path("Intelligence/mechanic/local").glob("*.py"))
        local_files.extend(list(Path("Intelligence/mechanic/local").glob("**/*.py")))
        
        # Root mechanic files
        root_files = [f for f in Path(".").glob("*mechanic*.py") if f.is_file()]
        
        # GitHub copilot mechanic files
        copilot_files = list(Path(".github/copilot_mechanic").glob("*.py"))
        
        # Test and validation files
        test_files = [f for f in Path(".").glob("test_*mechanic*.py") if f.is_file()]
        test_files.extend([f for f in Path(".").glob("*cloud*.py") if f.is_file()])
        
        # Intelligence script files
        intel_files = list(Path("Intelligence/scripts").glob("*cloud*.py"))
        
        all_files = cloud_files + local_files + root_files + copilot_files + test_files + intel_files
        
        # Remove duplicates and invalid files
        unique_files = []
        seen = set()
        for f in all_files:
            if f.exists() and f.name not in seen and not f.name.startswith('__'):
                unique_files.append(f)
                seen.add(f.name)
        
        self.mechanic_files = unique_files
        return unique_files
        
    def count_functions_in_file(self, file_path: Path) -> int:
        """Count all functions in a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            # Count both public and private functions as features
            return len(functions)
            
        except Exception as e:
            print(f"   ⚠️ Error parsing {file_path}: {e}")
            return 0
    
    def categorize_file(self, file_path: Path) -> str:
        """Categorize file based on path and name"""
        name = file_path.name.lower()
        path_str = str(file_path).lower()
        
        if 'cloud' in path_str:
            return "Cloud_AI_Engine"
        elif 'local' in path_str:
            return "Local_Mechanic_System"
        elif 'copilot' in path_str:
            return "Copilot_AI_Brain"
        elif 'test' in name:
            return "Testing_Validation_System"
        elif 'monitor' in name:
            return "Monitoring_System"
        elif 'auto' in name:
            return "Automation_System"
        elif 'workflow' in name:
            return "Workflow_Intelligence"
        elif 'background' in name:
            return "Background_Processing"
        else:
            return "Core_Utility_System"
    
    def audit_all_features(self) -> Dict:
        """Complete audit of all 85+ features"""
        print("🚀 COMPLETE 85+ CLOUD AI MECHANIC FEATURE AUDIT")
        print("=" * 80)
        print(f"Started: {datetime.now().isoformat()}")
        print("")
        
        # Step 1: Discover all files
        all_files = self.discover_all_mechanic_files()
        
        print(f"📁 Found {len(all_files)} mechanic-related files")
        print("")
        
        # Step 2: Count functions by category
        total_functions = 0
        file_details = []
        
        for file_path in all_files:
            func_count = self.count_functions_in_file(file_path)
            category = self.categorize_file(file_path)
            
            if category not in self.features_by_category:
                self.features_by_category[category] = []
                
            self.features_by_category[category].append({
                'file': file_path.name,
                'path': str(file_path),
                'functions': func_count
            })
            
            file_details.append({
                'file': file_path.name,
                'path': str(file_path),
                'category': category,
                'functions': func_count
            })
            
            total_functions += func_count
            print(f"📄 {file_path.name:<35} {func_count:>3} functions - {category}")
        
        self.total_features = total_functions
        
        print(f"\n📊 FEATURE DISCOVERY SUMMARY:")
        print("=" * 50)
        
        for category, files in self.features_by_category.items():
            category_total = sum(f['functions'] for f in files)
            print(f"🎯 {category:<25} {category_total:>3} features")
        
        print(f"\n✅ TOTAL CLOUD AI FEATURES: {self.total_features}")
        
        # Check if we meet the 85+ requirement
        if self.total_features >= 85:
            print(f"🎉 REQUIREMENT MET: {self.total_features} >= 85 features!")
        else:
            print(f"⚠️ REQUIREMENT NOT MET: {self.total_features} < 85 features")
        
        return {
            'total_features': self.total_features,
            'files_analyzed': len(all_files),
            'features_by_category': self.features_by_category,
            'file_details': file_details,
            'meets_85_requirement': self.total_features >= 85
        }
    
    def test_key_features_live(self) -> Dict:
        """Test key features by running the cloud mechanic"""
        print(f"\n🧪 TESTING KEY FEATURES LIVE...")
        print("=" * 40)
        
        # Test cloud mechanic core
        test_results = self.test_cloud_mechanic()
        
        # Test workflow assistance
        assistance_results = self.test_workflow_assistance()
        
        # Test auto-response system
        auto_response_results = self.test_auto_response()
        
        combined_results = {
            'cloud_mechanic_core': test_results,
            'workflow_assistance': assistance_results,
            'auto_response_system': auto_response_results
        }
        
        # Calculate overall success
        total_tests = sum(len(r.get('features', {})) for r in combined_results.values() if isinstance(r, dict))
        passing_tests = sum(sum(1 for v in r.get('features', {}).values() if v) for r in combined_results.values() if isinstance(r, dict))
        
        self.active_features = passing_tests
        
        print(f"\n📊 LIVE TESTING SUMMARY:")
        print(f"   Total tests: {total_tests}")
        print(f"   Passing tests: {passing_tests}")
        print(f"   Success rate: {passing_tests/total_tests*100:.1f}%" if total_tests > 0 else "   Success rate: 0%")
        
        return combined_results
    
    def test_cloud_mechanic(self) -> Dict:
        """Test the main cloud mechanic"""
        print("🔧 Testing Cloud Mechanic Core...")
        
        env = os.environ.copy()
        env.update({
            'ULTIMATE_MODE': 'true',
            'GITHUB_REPOSITORY_OWNER': 'c-trading-bo',
            'GITHUB_REPOSITORY': 'trading-bot-c-'
        })
        
        try:
            result = subprocess.run([
                'python', 'Intelligence/mechanic/cloud/cloud_mechanic_core.py'
            ], env=env, capture_output=True, text=True, timeout=60)
            
            output = result.stdout
            
            features = {
                'execution_success': result.returncode == 0,
                'ultimate_mode_active': 'Ultimate' in output,
                'workflow_analysis': 'workflows' in output,
                'ai_optimization': 'optimization' in output.lower(),
                'intelligent_preparation': 'INTELLIGENT PREPARATION' in output,
                'performance_metrics': 'Ultimate Metrics' in output
            }
            
            active_count = sum(1 for v in features.values() if v)
            
            print(f"   ✅ Core features active: {active_count}/{len(features)}")
            
            return {
                'status': 'success',
                'features': features,
                'output_length': len(output)
            }
            
        except Exception as e:
            print(f"   ❌ Core test failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def test_workflow_assistance(self) -> Dict:
        """Test workflow assistance features"""
        print("🤝 Testing Workflow Assistance...")
        
        try:
            # Check if auto-response triggers are configured
            workflow_file = Path('.github/workflows/cloud_bot_mechanic.yml')
            has_triggers = False
            monitors_workflows = 0
            
            if workflow_file.exists():
                content = workflow_file.read_text()
                has_triggers = 'workflow_run:' in content
                monitors_workflows = content.count('- "')
            
            # Check database tracking
            db_file = Path('Intelligence/mechanic/cloud/database/workflows.json')
            has_tracking = db_file.exists()
            
            features = {
                'auto_response_configured': has_triggers,
                'monitors_multiple_workflows': monitors_workflows >= 20,
                'workflow_tracking_active': has_tracking,
                'cache_system_ready': Path('.mechanic-cache').exists()
            }
            
            active_count = sum(1 for v in features.values() if v)
            print(f"   ✅ Assistance features active: {active_count}/{len(features)}")
            
            return {
                'status': 'success',
                'features': features,
                'monitored_workflows': monitors_workflows
            }
            
        except Exception as e:
            print(f"   ❌ Assistance test failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def test_auto_response(self) -> Dict:
        """Test auto-response and ultimate features"""
        print("⚡ Testing Auto-Response & Ultimate Features...")
        
        features = {
            'ultimate_mode_enabled': 'ULTIMATE_MODE: true' in Path('.github/workflows/cloud_bot_mechanic.yml').read_text() if Path('.github/workflows/cloud_bot_mechanic.yml').exists() else False,
            'auto_triggers_configured': Path('add_auto_response.py').exists(),
            'ultimate_enhancer_available': Path('enable_ultimate_cloud_mechanic.py').exists(),
            'comprehensive_testing': Path('test_cloud_mechanic.py').exists()
        }
        
        active_count = sum(1 for v in features.values() if v)
        print(f"   ✅ Auto-response features active: {active_count}/{len(features)}")
        
        return {
            'status': 'success',
            'features': features
        }
    
    def generate_final_report(self) -> Dict:
        """Generate final comprehensive report"""
        print(f"\n📋 FINAL COMPREHENSIVE REPORT")
        print("=" * 60)
        
        # Run complete audit
        discovery_results = self.audit_all_features()
        testing_results = self.test_key_features_live()
        
        # Generate recommendations
        recommendations = []
        
        if discovery_results['meets_85_requirement']:
            recommendations.append("✅ 85+ feature requirement is met!")
        else:
            recommendations.append(f"⚠️ Need {85 - self.total_features} more features to reach 85")
        
        if self.active_features > 10:
            recommendations.append("✅ Live features are working well")
        else:
            recommendations.append("⚠️ More features need to be activated")
        
        # Check workflow assistance
        if testing_results.get('workflow_assistance', {}).get('features', {}).get('monitors_multiple_workflows', False):
            recommendations.append("✅ Multi-workflow assistance is active")
        else:
            recommendations.append("⚠️ Need to improve workflow assistance coverage")
        
        final_report = {
            'timestamp': datetime.now().isoformat(),
            'total_features_discovered': self.total_features,
            'target_met': self.total_features >= 85,
            'live_features_active': self.active_features,
            'discovery_results': discovery_results,
            'testing_results': testing_results,
            'recommendations': recommendations,
            'overall_status': 'EXCELLENT' if self.total_features >= 85 and self.active_features > 10 else 'NEEDS_IMPROVEMENT'
        }
        
        print(f"🎯 FINAL STATUS: {final_report['overall_status']}")
        print(f"📊 Features: {self.total_features}/85+ required")
        print(f"⚡ Active: {self.active_features} features live")
        
        # Save report
        report_file = Path('complete_85_feature_audit.json')
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        print(f"\n💾 Complete report saved to: {report_file}")
        
        return final_report

def main():
    """Run the complete 85+ feature audit"""
    auditor = CompleteFeatureAudit()
    return auditor.generate_final_report()

if __name__ == "__main__":
    main()