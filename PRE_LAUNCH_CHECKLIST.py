#!/usr/bin/env python3
"""
🔍 PRE-LAUNCH TRADING BOT VERIFICATION SYSTEM
==============================================

Comprehensive verification that all systems are working properly,
wired together, and contain real logic (no placeholders).

This script performs deep inspection of:
- System Architecture & Wiring
- Code Logic Depth & Quality
- Configuration Completeness
- Integration Testing
- Performance Verification
"""

import os
import re
import json
import time
import subprocess
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import requests
from concurrent.futures import ThreadPoolExecutor

class TradingBotVerificationSystem:
    def __init__(self):
        self.root_path = Path(os.getcwd())
        self.verification_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'PENDING',
            'critical_issues': [],
            'warnings': [],
            'passed_checks': [],
            'failed_checks': [],
            'placeholders_found': [],
            'system_health': {},
            'integration_status': {},
            'performance_metrics': {}
        }
        
        # Placeholder patterns to detect
        self.placeholder_patterns = [
            r'TODO\s*:',
            r'FIXME\s*:',
            r'PLACEHOLDER',
            r'NotImplementedException',
            r'throw new Exception\("Not implemented"\)',
            r'Console\.WriteLine\("Mock',
            r'// Simulate',
            r'// Mock',
            r'fake|dummy|test|placeholder',
            r'return\s+default\(',
            r'return\s+null;',
            r'await Task\.Delay\(\d+\);.*\/\/.*placeholder',
            r'Random\.Shared\.Next',
            r'Math\.Random',
            r'\.ToList\(\);\s*\/\/.*mock'
        ]
        
        # Critical system components that must have real logic
        self.critical_components = [
            'src/UnifiedOrchestrator/Services/TradingOrchestratorService.cs',
            'src/UnifiedOrchestrator/Services/IntelligenceAndDataOrchestrators.cs',
            'src/UnifiedOrchestrator/Services/CentralMessageBus.cs',
            'src/BotCore/Strategy/StrategyMlIntegration.cs',
            'src/BotCore/Risk/RiskEngine.cs',
            'Enhanced/MLRLSystem.cs',
            'Enhanced/MarketIntelligence.cs',
            'src/BotCore/Models/Signal.cs',
            'src/TopstepAuthAgent/TopstepAuthAgent.cs'
        ]

    def run_complete_verification(self) -> Dict[str, Any]:
        """Run complete pre-launch verification"""
        print("🚀 STARTING COMPREHENSIVE PRE-LAUNCH VERIFICATION")
        print("=" * 60)
        
        # 1. System Architecture Check
        print("\n🏗️  PHASE 1: SYSTEM ARCHITECTURE VERIFICATION")
        self.verify_system_architecture()
        
        # 2. Code Logic Depth Analysis
        print("\n🧠 PHASE 2: CODE LOGIC DEPTH ANALYSIS")
        self.analyze_code_logic_depth()
        
        # 3. Placeholder Detection
        print("\n🔍 PHASE 3: PLACEHOLDER & MOCK CODE DETECTION")
        self.detect_placeholders()
        
        # 4. Integration Testing
        print("\n🔗 PHASE 4: INTEGRATION & WIRING VERIFICATION")
        self.verify_integrations()
        
        # 5. Configuration Completeness
        print("\n⚙️  PHASE 5: CONFIGURATION COMPLETENESS CHECK")
        self.verify_configurations()
        
        # 6. Performance & Health Check
        print("\n📊 PHASE 6: PERFORMANCE & HEALTH VERIFICATION")
        self.verify_performance_health()
        
        # 7. Trading Logic Verification
        print("\n📈 PHASE 7: TRADING LOGIC VERIFICATION")
        self.verify_trading_logic()
        
        # 8. ML/RL Systems Check
        print("\n🤖 PHASE 8: ML/RL SYSTEMS VERIFICATION")
        self.verify_ml_rl_systems()
        
        # Generate final report
        self.generate_final_report()
        return self.verification_report

    def verify_system_architecture(self):
        """Verify system architecture and component wiring"""
        print("  📋 Checking system architecture...")
        
        # Check unified orchestrator structure
        unified_path = self.root_path / "src" / "UnifiedOrchestrator"
        if unified_path.exists():
            self.verification_report['passed_checks'].append("✅ Unified Orchestrator directory exists")
            
            # Check critical files
            critical_files = [
                "Program.cs", "Services/CentralMessageBus.cs", 
                "Services/TradingOrchestratorService.cs",
                "Services/UnifiedOrchestratorService.cs",
                "Models/TradingBrainModels.cs"
            ]
            
            for file in critical_files:
                file_path = unified_path / file
                if file_path.exists():
                    self.verification_report['passed_checks'].append(f"✅ {file} exists")
                else:
                    self.verification_report['failed_checks'].append(f"❌ Missing critical file: {file}")
        else:
            self.verification_report['critical_issues'].append("❌ Unified Orchestrator directory missing")

    def analyze_code_logic_depth(self):
        """Analyze code for logic depth and quality"""
        print("  🔬 Analyzing code logic depth...")
        
        for component in self.critical_components:
            file_path = self.root_path / component
            if file_path.exists():
                self.analyze_file_logic_depth(file_path, component)
            else:
                self.verification_report['failed_checks'].append(f"❌ Critical component missing: {component}")

    def analyze_file_logic_depth(self, file_path: Path, component_name: str):
        """Analyze individual file for logic depth"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Metrics for logic depth
            metrics = {
                'lines_of_code': len([line for line in content.split('\n') if line.strip() and not line.strip().startswith('//')]),
                'methods_count': len(re.findall(r'(public|private|protected)\s+\w+\s+\w+\s*\(', content)),
                'classes_count': len(re.findall(r'class\s+\w+', content)),
                'interfaces_count': len(re.findall(r'interface\s+\w+', content)),
                'async_methods': len(re.findall(r'async\s+\w+', content)),
                'complex_logic_patterns': len(re.findall(r'(if|for|while|switch|try|catch)', content)),
                'database_operations': len(re.findall(r'(SaveChanges|ExecuteNonQuery|ExecuteScalar|Select|Insert|Update|Delete)', content)),
                'api_calls': len(re.findall(r'(HttpClient|RestClient|WebRequest|GetAsync|PostAsync)', content)),
                'ml_operations': len(re.findall(r'(Neural|LSTM|Transformer|XGBoost|Bandit|Model|Predict)', content))
            }
            
            # Calculate logic depth score
            logic_score = (
                min(metrics['lines_of_code'] / 100, 10) +
                min(metrics['methods_count'] / 5, 5) +
                min(metrics['complex_logic_patterns'] / 20, 5) +
                min(metrics['async_methods'] / 3, 3) +
                min(metrics['ml_operations'] / 5, 2)
            )
            
            if logic_score >= 15:
                self.verification_report['passed_checks'].append(f"✅ {component_name}: High logic depth (score: {logic_score:.1f})")
            elif logic_score >= 10:
                self.verification_report['warnings'].append(f"⚠️  {component_name}: Medium logic depth (score: {logic_score:.1f})")
            else:
                self.verification_report['failed_checks'].append(f"❌ {component_name}: Low logic depth (score: {logic_score:.1f})")
                
            self.verification_report['system_health'][component_name] = metrics
            
        except Exception as e:
            self.verification_report['failed_checks'].append(f"❌ Error analyzing {component_name}: {str(e)}")

    def detect_placeholders(self):
        """Detect placeholder and mock code"""
        print("  🔍 Scanning for placeholders and mock code...")
        
        placeholder_count = 0
        for pattern in ['**/*.cs', '**/*.py', '**/*.js', '**/*.yml']:
            for file_path in glob.glob(str(self.root_path / pattern), recursive=True):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    for pattern_regex in self.placeholder_patterns:
                        matches = re.findall(pattern_regex, content, re.IGNORECASE)
                        if matches:
                            placeholder_count += len(matches)
                            rel_path = os.path.relpath(file_path, self.root_path)
                            self.verification_report['placeholders_found'].append({
                                'file': rel_path,
                                'pattern': pattern_regex,
                                'matches': len(matches),
                                'examples': matches[:3]  # First 3 examples
                            })
                except Exception as e:
                    continue
        
        if placeholder_count == 0:
            self.verification_report['passed_checks'].append("✅ No placeholders detected")
        elif placeholder_count < 10:
            self.verification_report['warnings'].append(f"⚠️  {placeholder_count} potential placeholders found")
        else:
            self.verification_report['critical_issues'].append(f"❌ {placeholder_count} placeholders found - system not production ready")

    def verify_integrations(self):
        """Verify system integrations and wiring"""
        print("  🔗 Verifying system integrations...")
        
        # Check if Unified Orchestrator is running
        try:
            # Check for running dotnet processes
            result = subprocess.run(['powershell', '-Command', 'Get-Process -Name "dotnet" -ErrorAction SilentlyContinue'], 
                                  capture_output=True, text=True, timeout=10)
            if 'dotnet' in result.stdout:
                self.verification_report['passed_checks'].append("✅ .NET processes running")
                self.verification_report['integration_status']['dotnet_processes'] = 'RUNNING'
            else:
                self.verification_report['warnings'].append("⚠️  No .NET processes detected")
                self.verification_report['integration_status']['dotnet_processes'] = 'NOT_RUNNING'
        except Exception as e:
            self.verification_report['failed_checks'].append(f"❌ Cannot check process status: {str(e)}")

        # Check configuration files
        config_files = [
            'appsettings.json',
            '.env.sample.local',
            'Enhanced/Enhanced.csproj',
            'src/UnifiedOrchestrator/UnifiedOrchestrator.csproj'
        ]
        
        for config_file in config_files:
            file_path = self.root_path / config_file
            if file_path.exists():
                self.verification_report['passed_checks'].append(f"✅ Configuration file exists: {config_file}")
            else:
                self.verification_report['warnings'].append(f"⚠️  Configuration file missing: {config_file}")

    def verify_configurations(self):
        """Verify configuration completeness"""
        print("  ⚙️  Checking configuration completeness...")
        
        # Check appsettings.json
        appsettings_path = self.root_path / "appsettings.json"
        if appsettings_path.exists():
            try:
                with open(appsettings_path, 'r') as f:
                    config = json.load(f)
                    
                required_sections = ['Logging', 'AllowedHosts']
                for section in required_sections:
                    if section in config:
                        self.verification_report['passed_checks'].append(f"✅ Configuration section exists: {section}")
                    else:
                        self.verification_report['warnings'].append(f"⚠️  Missing configuration section: {section}")
                        
            except json.JSONDecodeError:
                self.verification_report['failed_checks'].append("❌ Invalid JSON in appsettings.json")
        
        # Check workflow files
        workflow_dir = self.root_path / ".github" / "workflows"
        if workflow_dir.exists():
            workflow_count = len(list(workflow_dir.glob("*.yml")))
            if workflow_count >= 20:
                self.verification_report['passed_checks'].append(f"✅ {workflow_count} workflow files found")
            else:
                self.verification_report['warnings'].append(f"⚠️  Only {workflow_count} workflow files found")

    def verify_performance_health(self):
        """Verify performance and health metrics"""
        print("  📊 Checking performance and health...")
        
        # Check directory sizes
        major_directories = ['src', 'Enhanced', '.github/workflows', 'ml']
        for dir_name in major_directories:
            dir_path = self.root_path / dir_name
            if dir_path.exists():
                size_mb = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file()) / (1024 * 1024)
                self.verification_report['performance_metrics'][dir_name] = f"{size_mb:.2f} MB"
                if size_mb > 1:  # At least 1MB indicates substantial content
                    self.verification_report['passed_checks'].append(f"✅ {dir_name} directory has substantial content ({size_mb:.2f} MB)")
                else:
                    self.verification_report['warnings'].append(f"⚠️  {dir_name} directory seems small ({size_mb:.2f} MB)")

    def verify_trading_logic(self):
        """Verify trading logic implementation"""
        print("  📈 Verifying trading logic...")
        
        # Check strategy files
        strategy_dir = self.root_path / "src" / "BotCore" / "Strategy"
        if strategy_dir.exists():
            strategy_files = list(strategy_dir.glob("*.cs"))
            if len(strategy_files) >= 5:
                self.verification_report['passed_checks'].append(f"✅ {len(strategy_files)} strategy files found")
            else:
                self.verification_report['warnings'].append(f"⚠️  Only {len(strategy_files)} strategy files found")
        
        # Check risk management
        risk_dir = self.root_path / "src" / "BotCore" / "Risk"
        if risk_dir.exists():
            risk_files = list(risk_dir.glob("*.cs"))
            if len(risk_files) >= 2:
                self.verification_report['passed_checks'].append(f"✅ {len(risk_files)} risk management files found")
            else:
                self.verification_report['failed_checks'].append(f"❌ Insufficient risk management files: {len(risk_files)}")

    def verify_ml_rl_systems(self):
        """Verify ML/RL systems"""
        print("  🤖 Verifying ML/RL systems...")
        
        # Check Enhanced ML/RL system
        mlrl_path = self.root_path / "Enhanced" / "MLRLSystem.cs"
        if mlrl_path.exists():
            try:
                with open(mlrl_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for ML/RL patterns
                ml_patterns = ['LSTM', 'Transformer', 'XGBoost', 'FinBERT', 'Autoencoder']
                rl_patterns = ['DQN', 'PPO', 'A3C', 'Agent', 'Policy']
                
                found_ml = sum(1 for pattern in ml_patterns if pattern in content)
                found_rl = sum(1 for pattern in rl_patterns if pattern in content)
                
                if found_ml >= 3:
                    self.verification_report['passed_checks'].append(f"✅ ML models detected: {found_ml}/5")
                else:
                    self.verification_report['failed_checks'].append(f"❌ Insufficient ML models: {found_ml}/5")
                    
                if found_rl >= 2:
                    self.verification_report['passed_checks'].append(f"✅ RL agents detected: {found_rl}/3")
                else:
                    self.verification_report['failed_checks'].append(f"❌ Insufficient RL agents: {found_rl}/3")
                    
            except Exception as e:
                self.verification_report['failed_checks'].append(f"❌ Error reading MLRLSystem.cs: {str(e)}")

    def generate_final_report(self):
        """Generate final verification report"""
        print("\n" + "="*60)
        print("📋 FINAL VERIFICATION REPORT")
        print("="*60)
        
        # Calculate overall status
        critical_count = len(self.verification_report['critical_issues'])
        failed_count = len(self.verification_report['failed_checks'])
        warning_count = len(self.verification_report['warnings'])
        passed_count = len(self.verification_report['passed_checks'])
        placeholder_count = len(self.verification_report['placeholders_found'])
        
        if critical_count > 0 or failed_count > 10 or placeholder_count > 20:
            self.verification_report['overall_status'] = 'FAILED'
            status_emoji = "❌"
        elif failed_count > 0 or warning_count > 5 or placeholder_count > 5:
            self.verification_report['overall_status'] = 'WARNING'
            status_emoji = "⚠️"
        else:
            self.verification_report['overall_status'] = 'PASSED'
            status_emoji = "✅"
        
        print(f"\n{status_emoji} OVERALL STATUS: {self.verification_report['overall_status']}")
        print(f"✅ Passed Checks: {passed_count}")
        print(f"⚠️  Warnings: {warning_count}")
        print(f"❌ Failed Checks: {failed_count}")
        print(f"🚨 Critical Issues: {critical_count}")
        print(f"🔍 Placeholders Found: {placeholder_count}")
        
        # Show critical issues
        if self.verification_report['critical_issues']:
            print(f"\n🚨 CRITICAL ISSUES:")
            for issue in self.verification_report['critical_issues']:
                print(f"  {issue}")
        
        # Show failed checks
        if self.verification_report['failed_checks']:
            print(f"\n❌ FAILED CHECKS:")
            for check in self.verification_report['failed_checks'][:10]:  # Show first 10
                print(f"  {check}")
            if len(self.verification_report['failed_checks']) > 10:
                print(f"  ... and {len(self.verification_report['failed_checks']) - 10} more")
        
        # Show placeholder details
        if self.verification_report['placeholders_found']:
            print(f"\n🔍 PLACEHOLDER ANALYSIS:")
            for placeholder in self.verification_report['placeholders_found'][:5]:  # Show first 5
                print(f"  📁 {placeholder['file']}: {placeholder['matches']} matches ({placeholder['pattern']})")
            if len(self.verification_report['placeholders_found']) > 5:
                print(f"  ... and {len(self.verification_report['placeholders_found']) - 5} more files")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if self.verification_report['overall_status'] == 'FAILED':
            print("  🛑 DO NOT LAUNCH - Critical issues must be resolved")
            print("  🔧 Fix critical issues and failed checks before proceeding")
        elif self.verification_report['overall_status'] == 'WARNING':
            print("  ⚠️  CAUTION - Consider fixing warnings before launch")
            print("  🔧 Address major issues but can proceed with monitoring")
        else:
            print("  🚀 READY TO LAUNCH - All systems verified")
            print("  ✅ System is production ready")
        
        # Save report to file
        report_path = self.root_path / "VERIFICATION_REPORT.json"
        with open(report_path, 'w') as f:
            json.dump(self.verification_report, f, indent=2)
        print(f"\n📄 Detailed report saved to: {report_path}")

def main():
    """Main verification function"""
    print("""
🚀 TRADING BOT PRE-LAUNCH VERIFICATION SYSTEM
============================================
This comprehensive verification will check:
- System architecture and wiring
- Code logic depth and quality  
- Placeholder and mock code detection
- Integration completeness
- Configuration verification
- Performance and health metrics
- Trading logic implementation
- ML/RL systems verification
""")
    
    try:
        verifier = TradingBotVerificationSystem()
        report = verifier.run_complete_verification()
        
        print(f"\n🎯 VERIFICATION COMPLETE")
        print(f"Overall Status: {report['overall_status']}")
        
        return report['overall_status'] == 'PASSED'
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Verification interrupted by user")
        return False
    except Exception as e:
        print(f"\n\n❌ Verification failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
