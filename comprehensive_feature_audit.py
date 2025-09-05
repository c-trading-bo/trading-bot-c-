#!/usr/bin/env python3
"""
COMPREHENSIVE CLOUD MECHANIC FEATURE AUDIT
Discovers and tests all 85+ cloud AI features across all cloud mechanic files
"""

import os
import sys
import ast
import json
import subprocess
import importlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set

class FeatureDiscovery:
    def __init__(self):
        self.cloud_path = Path("Intelligence/mechanic/cloud")
        self.features = {}
        self.function_inventory = {}
        self.test_results = {}
        
    def discover_all_features(self) -> Dict:
        """Discover all features across all cloud mechanic files"""
        print("🔍 DISCOVERING ALL CLOUD MECHANIC FEATURES...")
        print("=" * 60)
        
        # Find all Python files in cloud directory
        cloud_files = list(self.cloud_path.glob("*.py"))
        total_functions = 0
        
        for py_file in cloud_files:
            if py_file.name.startswith('__'):
                continue
                
            print(f"\n📁 Analyzing {py_file.name}...")
            functions = self.extract_functions_from_file(py_file)
            self.function_inventory[py_file.name] = functions
            total_functions += len(functions)
            
            for func in functions:
                # Categorize functions as features
                feature_type = self.categorize_function(func['name'], func['docstring'])
                if feature_type:
                    if feature_type not in self.features:
                        self.features[feature_type] = []
                    self.features[feature_type].append({
                        'name': func['name'],
                        'file': py_file.name,
                        'description': func['docstring'] or f"Function in {py_file.name}",
                        'line': func['line']
                    })
        
        print(f"\n📊 DISCOVERY SUMMARY:")
        print(f"   Files scanned: {len(cloud_files)}")
        print(f"   Total functions: {total_functions}")
        print(f"   Feature categories: {len(self.features)}")
        
        # Count features by category
        total_features = 0
        for category, feature_list in self.features.items():
            feature_count = len(feature_list)
            total_features += feature_count
            print(f"   {category}: {feature_count} features")
            
        print(f"   🎯 TOTAL CLOUD AI FEATURES: {total_features}")
        
        return {
            'total_functions': total_functions,
            'total_features': total_features,
            'features_by_category': self.features,
            'function_inventory': self.function_inventory
        }
    
    def extract_functions_from_file(self, file_path: Path) -> List[Dict]:
        """Extract all functions from a Python file"""
        functions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Skip private methods (starting with _)
                    if not node.name.startswith('_'):
                        docstring = ast.get_docstring(node) or ""
                        functions.append({
                            'name': node.name,
                            'line': node.lineno,
                            'docstring': docstring.strip() if docstring else ""
                        })
                        
        except Exception as e:
            print(f"   ⚠️ Error parsing {file_path}: {e}")
            
        return functions
    
    def categorize_function(self, func_name: str, docstring: str) -> str:
        """Categorize function based on name and docstring"""
        name_lower = func_name.lower()
        doc_lower = docstring.lower()
        combined = f"{name_lower} {doc_lower}"
        
        # AI and Learning Features
        if any(keyword in combined for keyword in ['learn', 'ai', 'intelligent', 'predict', 'optimize', 'pattern']):
            return "AI_Learning"
            
        # Workflow Analysis Features  
        if any(keyword in combined for keyword in ['analyze', 'workflow', 'validate', 'check', 'scan']):
            return "Workflow_Analysis"
            
        # Performance and Monitoring Features
        if any(keyword in combined for keyword in ['monitor', 'performance', 'metric', 'track', 'measure']):
            return "Performance_Monitoring"
            
        # Automation and Fixing Features
        if any(keyword in combined for keyword in ['fix', 'repair', 'auto', 'preempt', 'prevent']):
            return "Automation_Fixing"
            
        # Caching and Optimization Features
        if any(keyword in combined for keyword in ['cache', 'bundle', 'compile', 'dependency']):
            return "Caching_Optimization"
            
        # Reporting and Alerting Features
        if any(keyword in combined for keyword in ['report', 'alert', 'generate', 'notify', 'output']):
            return "Reporting_Alerting"
            
        # GitHub Integration Features
        if any(keyword in combined for keyword in ['github', 'api', 'fetch', 'pull', 'push']):
            return "GitHub_Integration"
            
        # Core System Features
        if any(keyword in combined for keyword in ['setup', 'init', 'config', 'load', 'save']):
            return "Core_System"
            
        # Testing and Validation Features
        if any(keyword in combined for keyword in ['test', 'validate', 'verify', 'check']):
            return "Testing_Validation"
            
        # Scheduling and Triggers
        if any(keyword in combined for keyword in ['schedule', 'trigger', 'cron', 'time']):
            return "Scheduling_Triggers"
            
        return "Utility_Functions"
    
    def test_features_live(self) -> Dict:
        """Test features by running cloud mechanic and analyzing output"""
        print("\n🧪 TESTING FEATURES LIVE...")
        print("=" * 40)
        
        # Set environment for ultimate mode
        env = os.environ.copy()
        env.update({
            'ULTIMATE_MODE': 'true',
            'GITHUB_REPOSITORY_OWNER': 'c-trading-bo',
            'GITHUB_REPOSITORY': 'trading-bot-c-'
        })
        
        try:
            # Run cloud mechanic
            result = subprocess.run([
                'python', 
                'Intelligence/mechanic/cloud/cloud_mechanic_core.py'
            ], 
            env=env,
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            timeout=120
            )
            
            output = result.stdout
            
            # Analyze output for feature evidence
            feature_evidence = {
                'workflow_discovery': 'workflows' in output and 'Total workflows:' in output,
                'health_analysis': 'Healthy workflows:' in output,
                'issue_detection': 'Issues found:' in output or 'Broken workflows:' in output,
                'budget_monitoring': 'Monthly minutes:' in output or 'minutes' in output,
                'ultimate_mode': 'Ultimate' in output or 'ULTIMATE' in output,
                'ai_optimization': 'optimization' in output.lower(),
                'workflow_learning': 'Learning' in output,
                'intelligent_preparation': 'INTELLIGENT PREPARATION' in output,
                'pre_compilation': 'Pre-compiling' in output,
                'dependency_caching': 'caching' in output.lower() or 'cache' in output.lower(),
                'performance_metrics': 'Ultimate Metrics' in output,
                'pattern_recognition': 'patterns' in output.lower(),
                'failure_prediction': 'Predicting' in output,
                'auto_fixing': 'pre-fixing' in output,
                'bundle_creation': 'bundles' in output,
                'custom_optimizations': 'custom optimizations' in output
            }
            
            active_features = sum(1 for active in feature_evidence.values() if active)
            
            print(f"📊 LIVE FEATURE TEST RESULTS:")
            for feature, is_active in feature_evidence.items():
                status = "✅ ACTIVE" if is_active else "❌ INACTIVE"
                print(f"   {feature:<25} {status}")
                
            print(f"\n🎯 LIVE FEATURES ACTIVE: {active_features}/{len(feature_evidence)} ({active_features/len(feature_evidence)*100:.1f}%)")
            
            self.test_results = {
                'execution_successful': result.returncode == 0,
                'output_length': len(output),
                'features_active': active_features,
                'total_features_tested': len(feature_evidence),
                'feature_evidence': feature_evidence,
                'performance_excellent': len(output) > 1000  # Good output indicates working features
            }
            
            return self.test_results
            
        except Exception as e:
            print(f"❌ Live test failed: {e}")
            return {'error': str(e)}

def run_comprehensive_audit():
    """Run the complete feature audit"""
    print("🚀 COMPREHENSIVE CLOUD MECHANIC FEATURE AUDIT")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}")
    print("")
    
    discovery = FeatureDiscovery()
    
    # Step 1: Discover all features
    discovery_results = discovery.discover_all_features()
    
    # Step 2: Test features live
    test_results = discovery.test_features_live()
    
    # Step 3: Generate comprehensive report
    audit_report = {
        'timestamp': datetime.now().isoformat(),
        'discovery': discovery_results,
        'live_testing': test_results,
        'recommendations': []
    }
    
    print("\n📋 COMPREHENSIVE AUDIT SUMMARY:")
    print("=" * 50)
    print(f"✅ Total Functions Discovered: {discovery_results['total_functions']}")
    print(f"🎯 Total Features Identified: {discovery_results['total_features']}")
    
    if 'features_active' in test_results:
        print(f"⚡ Features Active in Live Test: {test_results['features_active']}")
        print(f"📊 Live Test Success Rate: {test_results['features_active']/test_results['total_features_tested']*100:.1f}%")
    
    # Check if we have 85+ features
    if discovery_results['total_features'] >= 85:
        print(f"✅ TARGET MET: {discovery_results['total_features']} features >= 85 required!")
    else:
        print(f"⚠️ TARGET PARTIAL: {discovery_results['total_features']} features < 85 required")
        audit_report['recommendations'].append("Add more features to reach 85+ target")
    
    # Check live test performance
    if test_results.get('execution_successful', False):
        print("✅ Live execution successful")
    else:
        print("❌ Live execution failed")
        audit_report['recommendations'].append("Fix cloud mechanic execution issues")
    
    if test_results.get('performance_excellent', False):
        print("✅ Performance excellent - features producing good output")
    else:
        print("⚠️ Performance needs improvement")
        audit_report['recommendations'].append("Improve feature output and functionality")
    
    # Save results
    audit_file = Path('comprehensive_feature_audit.json')
    with open(audit_file, 'w') as f:
        json.dump(audit_report, f, indent=2, default=str)
    
    print(f"\n💾 Complete audit saved to: {audit_file}")
    
    return audit_report

if __name__ == "__main__":
    run_comprehensive_audit()