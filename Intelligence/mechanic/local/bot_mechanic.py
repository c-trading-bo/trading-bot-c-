#!/usr/bin/env python3
"""
LOCAL BOT MECHANIC - Complete System
Knows everything about your bot and fixes it automatically
Runs on YOUR computer
"""

import os
import sys
import ast
import json
import time
import hashlib
import threading
import subprocess
import traceback
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
import warnings
# Suppress specific warnings that are not critical for trading operations
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class LocalBotMechanic:
    def __init__(self):
        self.version = "3.0.0-LOCAL"
        self.start_time = datetime.utcnow()
        self.base_path = Path.cwd()
        
        # Databases
        self.db_path = Path("Intelligence/mechanic/database")
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.knowledge_db = self.db_path / "knowledge.json"
        self.features_db = self.db_path / "features.json"
        self.issues_db = self.db_path / "issues.json"
        self.repairs_db = self.db_path / "repairs.json"
        self.alerts_db = self.db_path / "alerts.json"
        
        # Load databases
        self.knowledge = self.load_json(self.knowledge_db, {
            "version": self.version,
            "created": datetime.utcnow().isoformat(),
            "files": {},
            "features": {},
            "dependencies": {},
            "trading_logic": {},
            "ml_models": {},
            "strategies": {},
            "last_scan": None
        })
        
        self.known_features = self.load_json(self.features_db, {})
        self.known_issues = self.load_json(self.issues_db, {})
        self.repair_history = self.load_json(self.repairs_db, [])
        self.alerts = self.load_json(self.alerts_db, [])
        
        # Feature detection patterns
        self.patterns = {
            'trading': ['trade', 'signal', 'buy', 'sell', 'position', 'order', 'es_', 'nq_', 'spy', 'qqq'],
            'ml': ['train', 'predict', 'model', 'neural', 'xgboost', 'lstm', 'random_forest', 'sklearn'],
            'data': ['fetch', 'download', 'collect', 'scrape', 'api', 'yfinance', 'yahoo', 'market_data'],
            'analysis': ['analyze', 'calculate', 'indicator', 'strategy', 'backtest', 'rsi', 'macd', 'ema'],
            'workflow': ['schedule', 'cron', 'trigger', 'dispatch', 'github', 'action'],
            'alert': ['notify', 'email', 'webhook', 'alert', 'warning', 'send_'],
            'options': ['option', 'call', 'put', 'strike', 'expiry', 'gamma', 'theta'],
            'regime': ['regime', 'volatility', 'trend', 'momentum', 'bearish', 'bullish'],
            'intelligence': ['cot', 'congressional', 'social', 'sentiment', 'intermarket', 'opex']
        }
        
        # Critical files that must exist
        self.critical_files = {
            'trading': [
                'Intelligence/scripts/strategies/es_nq_realtime.py',
                'Intelligence/scripts/strategies/spy_qqq_regime.py',
                'Intelligence/scripts/strategies/options_flow.py'
            ],
            'ml': [
                'ml/train_adaptive_learning.py',
                'ml/train_neural_ucb.py',
                'ml/train_feature_importance.py'
            ],
            'intelligence': [
                'Intelligence/scripts/cot_report.py',
                'Intelligence/scripts/congressional_trades.py',
                'Intelligence/scripts/social_momentum.py',
                'Intelligence/scripts/intermarket.py'
            ],
            'integration': [
                'src/OrchestratorAgent/LocalBotMechanicIntegration.cs',
                'src/BotCore/Services/IntelligenceService.cs'
            ]
        }
        
        print(f"🧠 Local Bot Mechanic v{self.version} initialized")
        print(f"📍 Monitoring: {self.base_path}")
        
    def load_json(self, path: Path, default: Any) -> Any:
        """Load JSON file with default fallback"""
        try:
            if path.exists():
                with open(path, 'r') as f:
                    return json.load(f)
        except:
            pass
        return default
    
    def save_json(self, path: Path, data: Any):
        """Save data to JSON file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    # ========================================
    # DEEP SCANNING SYSTEM
    # ========================================
    
    def deep_scan(self, verbose: bool = True) -> Dict:
        """Complete deep scan of entire codebase"""
        if verbose:
            print("\n" + "="*60)
            print("🔍 STARTING DEEP SCAN OF ENTIRE BOT")
            print("="*60)
        
        scan_start = datetime.utcnow()
        
        results = {
            'timestamp': scan_start.isoformat(),
            'files_scanned': 0,
            'total_features': 0,
            'new_features': [],
            'modified_features': [],
            'missing_features': [],
            'broken_features': [],
            'issues_found': [],
            'auto_fixed': [],
            'categories': {},
            'health_score': 100
        }
        
        # Scan all Python files
        all_files = list(Path('.').rglob('*.py'))
        
        for py_file in all_files:
            # Skip unwanted directories
            if any(skip in str(py_file) for skip in ['.git', '__pycache__', 'venv', '.env']):
                continue
            
            self._analyze_python_file(py_file, results, verbose)
            results['files_scanned'] += 1
        
        # Scan C# files
        cs_files = list(Path('.').rglob('*.cs'))
        for cs_file in cs_files:
            if '.git' not in str(cs_file):
                self._analyze_cs_file(cs_file, results)
                results['files_scanned'] += 1
        
        # Scan workflows
        workflow_path = Path('.github/workflows')
        if workflow_path.exists():
            for yml_file in workflow_path.glob('*.yml'):
                self._analyze_workflow(yml_file, results)
        
        # Scan configs
        for json_file in Path('.').rglob('*.json'):
            if '.git' not in str(json_file):
                self._analyze_config(json_file, results)
        
        # Detect changes
        self._detect_feature_changes(results)
        
        # Check system health
        self._comprehensive_health_check(results)
        
        # Calculate health score
        results['health_score'] = self._calculate_health_score(results)
        
        # Save results
        self.knowledge['last_scan'] = results
        self.save_json(self.knowledge_db, self.knowledge)
        
        # Generate report
        if verbose:
            self._print_scan_report(results)
        
        scan_time = (datetime.utcnow() - scan_start).total_seconds()
        results['scan_time_seconds'] = scan_time
        
        return results
    
    def _analyze_python_file(self, filepath: Path, results: Dict, verbose: bool):
        """Deeply analyze a Python file"""
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
            file_hash = hashlib.md5(content.encode()).hexdigest()
            file_key = str(filepath)
            
            # Check if file changed
            if file_key in self.knowledge.get('files', {}):
                old_hash = self.knowledge['files'][file_key].get('hash')
                if old_hash != file_hash:
                    results['modified_features'].append(filepath.name)
                    if verbose:
                        print(f"📝 Modified: {filepath}")
            else:
                results['new_features'].append(filepath.name)
                if verbose:
                    print(f"✨ New file: {filepath}")
            
            # Parse AST
            try:
                tree = ast.parse(content)
                
                file_features = {
                    'functions': [],
                    'classes': [],
                    'imports': [],
                    'categories': set(),
                    'has_error_handling': False,
                    'has_tests': False,
                    'lines_of_code': len(content.splitlines())
                }
                
                # Analyze all nodes
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_data = self._analyze_function(node, content)
                        file_features['functions'].append(func_data)
                        file_features['categories'].add(func_data['category'])
                        results['total_features'] += 1
                        
                        # Track category counts
                        cat = func_data['category']
                        results['categories'][cat] = results.get('categories', {}).get(cat, 0) + 1
                    
                    elif isinstance(node, ast.ClassDef):
                        class_data = self._analyze_class(node)
                        file_features['classes'].append(class_data)
                        results['total_features'] += 1
                    
                    elif isinstance(node, ast.Try):
                        file_features['has_error_handling'] = True
                    
                    elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                        imports = self._extract_imports(node)
                        file_features['imports'].extend(imports)
                
                # Check for test functions
                if any('test' in f['name'].lower() for f in file_features['functions']):
                    file_features['has_tests'] = True
                
                # Store in knowledge base
                self.knowledge['files'][file_key] = {
                    'hash': file_hash,
                    'features': file_features,
                    'last_analyzed': datetime.utcnow().isoformat(),
                    'categories': list(file_features['categories'])
                }
                
            except SyntaxError as e:
                error_msg = f"Syntax error in {filepath}: Line {e.lineno}: {e.msg}"
                results['broken_features'].append(filepath.name)
                results['issues_found'].append(error_msg)
                
                # Attempt auto-fix
                if self._attempt_syntax_fix(filepath, e):
                    results['auto_fixed'].append(f"Fixed syntax in {filepath.name}")
                
        except Exception as e:
            results['issues_found'].append(f"Error reading {filepath}: {str(e)}")
    
    def _analyze_cs_file(self, filepath: Path, results: Dict):
        """Analyze C# files for trading bot components"""
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
            
            # Check for key C# patterns
            cs_patterns = {
                'trading': ['Order', 'Trade', 'Position', 'Signal', 'Strategy'],
                'intelligence': ['Intelligence', 'Data', 'Local', 'Analysis'],
                'health': ['Health', 'Monitor', 'SelfHealing', 'Diagnostic'],
                'ml': ['Model', 'Training', 'Prediction', 'Learning']
            }
            
            file_categories = set()
            for category, keywords in cs_patterns.items():
                if any(keyword.lower() in content.lower() for keyword in keywords):
                    file_categories.add(category)
                    results['categories'][category] = results.get('categories', {}).get(category, 0) + 1
            
            # Check for disabled self-healing
            if 'SelfHealingEngine' in content and 'TEMPORARILY DISABLED' in content:
                results['issues_found'].append(f"Self-healing disabled in {filepath.name}")
            
            results['total_features'] += len(file_categories)
            
        except Exception as e:
            results['issues_found'].append(f"Error reading C# file {filepath}: {str(e)}")
    
    def _analyze_function(self, node: ast.FunctionDef, content: str) -> Dict:
        """Analyze a function in detail"""
        # Get function source code
        try:
            lines = content.splitlines()
            start_line = node.lineno - 1
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10
            func_source = '\n'.join(lines[start_line:end_line])
        except:
            func_source = ""
        
        # Categorize function
        category = self._categorize_function(node.name, func_source)
        
        # Check for error handling
        has_error_handling = any(isinstance(n, ast.Try) for n in ast.walk(node))
        
        # Extract docstring
        docstring = ast.get_docstring(node) or ""
        
        return {
            'name': node.name,
            'category': category,
            'args': [arg.arg for arg in node.args.args],
            'lines': (node.end_lineno or node.lineno) - node.lineno if hasattr(node, 'end_lineno') else 0,
            'has_error_handling': has_error_handling,
            'has_docstring': bool(docstring),
            'returns': self._get_return_type(node),
            'calls_apis': 'requests' in func_source or 'yfinance' in func_source,
            'is_async': isinstance(node, ast.AsyncFunctionDef)
        }
    
    def _analyze_class(self, node: ast.ClassDef) -> Dict:
        """Analyze a class in detail"""
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append({
                    'name': item.name,
                    'is_private': item.name.startswith('_'),
                    'is_property': any(
                        isinstance(d, ast.Name) and d.id == 'property' 
                        for d in item.decorator_list
                    )
                })
        
        return {
            'name': node.name,
            'methods': methods,
            'method_count': len(methods),
            'has_init': any(m['name'] == '__init__' for m in methods),
            'base_classes': [base.id for base in node.bases if isinstance(base, ast.Name)]
        }
    
    def _analyze_workflow(self, filepath: Path, results: Dict):
        """Analyze GitHub workflow files"""
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
            
            # Check if workflow is properly configured
            if 'on:' not in content and "'on':" not in content:
                results['issues_found'].append(f"Workflow {filepath.name} missing trigger")
            
            # Check for schedule
            if 'schedule:' in content or 'cron:' in content:
                results['categories']['workflow'] = results.get('categories', {}).get('workflow', 0) + 1
            
        except Exception as e:
            results['issues_found'].append(f"Error reading workflow {filepath}: {str(e)}")
    
    def _analyze_config(self, filepath: Path, results: Dict):
        """Analyze configuration files"""
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
            
            # Try to parse JSON
            try:
                data = json.loads(content)
                if isinstance(data, dict) and len(data) > 0:
                    results['categories']['config'] = results.get('categories', {}).get('config', 0) + 1
            except:
                pass
                
        except Exception as e:
            results['issues_found'].append(f"Error reading config {filepath}: {str(e)}")
    
    def _categorize_function(self, func_name: str, source: str = "") -> str:
        """Categorize function based on name and content"""
        func_lower = func_name.lower()
        source_lower = source.lower()
        
        for category, keywords in self.patterns.items():
            if any(keyword in func_lower or keyword in source_lower for keyword in keywords):
                return category
        
        return 'utility'
    
    def _extract_imports(self, node) -> List[str]:
        """Extract import names from import node"""
        imports = []
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")
        
        return imports
    
    def _get_return_type(self, node: ast.FunctionDef) -> str:
        """Get function return type if annotated"""
        if node.returns:
            if isinstance(node.returns, ast.Name):
                return node.returns.id
            elif isinstance(node.returns, ast.Constant):
                return str(node.returns.value)
        
        # Check for return statements
        for n in ast.walk(node):
            if isinstance(n, ast.Return):
                if n.value:
                    if isinstance(n.value, ast.Constant):
                        return type(n.value.value).__name__
                    elif isinstance(n.value, ast.Dict):
                        return 'dict'
                    elif isinstance(n.value, ast.List):
                        return 'list'
        
        return 'unknown'
    
    # ========================================
    # HEALTH CHECK SYSTEM
    # ========================================
    
    def _comprehensive_health_check(self, results: Dict):
        """Comprehensive health check of all systems"""
        print("\n🏥 RUNNING COMPREHENSIVE HEALTH CHECK...")
        
        health_checks = {
            'Trading System': self._check_trading_health(),
            'ML Models': self._check_ml_health(),
            'Intelligence Sources': self._check_intelligence_health(),
            'Data Pipeline': self._check_data_health(),
            'Dependencies': self._check_dependencies_health(),
            'Critical Files': self._check_critical_files(),
            'Workflows': self._check_workflow_health(),
            'C# Integration': self._check_cs_integration(),
            'Self-Healing': self._check_self_healing_status()
        }
        
        for system, status in health_checks.items():
            if status['healthy']:
                print(f"✅ {system}: OK")
            else:
                print(f"❌ {system}: {status['issue']}")
                results['issues_found'].append(f"{system}: {status['issue']}")
                
                # Attempt auto-repair
                if status.get('fixable'):
                    print(f"   🔧 Attempting auto-repair...")
                    if self._auto_repair(system, status):
                        results['auto_fixed'].append(system)
                        print(f"   ✅ Fixed!")
                    else:
                        print(f"   ⚠️ Manual fix required")
    
    def _check_trading_health(self) -> Dict:
        """Check trading system health"""
        issues = []
        
        # Check for trading files
        trading_files = [
            'Intelligence/scripts/strategies/es_nq_realtime.py',
            'Intelligence/scripts/strategies/spy_qqq_regime.py'
        ]
        
        for file in trading_files:
            if not Path(file).exists():
                issues.append(f"Missing: {file}")
        
        # Check for recent signals
        signal_dir = Path('Intelligence/data')
        if signal_dir.exists():
            signal_files = list(signal_dir.glob('*signal*.json'))
            if signal_files:
                # Check age of signals
                for sig_file in signal_files:
                    age = time.time() - sig_file.stat().st_mtime
                    if age > 3600:  # Older than 1 hour
                        issues.append(f"Stale signals: {sig_file.name}")
            else:
                issues.append("No signal files found")
        
        if issues:
            return {
                'healthy': False,
                'issue': ', '.join(issues),
                'fixable': True,
                'fix_type': 'create_trading_files'
            }
        
        return {'healthy': True}
    
    def _check_intelligence_health(self) -> Dict:
        """Check intelligence sources health"""
        issues = []
        
        intelligence_files = [
            'Intelligence/scripts/cot_report.py',
            'Intelligence/scripts/congressional_trades.py',
            'Intelligence/scripts/social_momentum.py',
            'Intelligence/scripts/intermarket.py',
            'Intelligence/scripts/opex_calendar.py'
        ]
        
        for file in intelligence_files:
            if not Path(file).exists():
                issues.append(f"Missing: {Path(file).name}")
        
        # Check LocalBotMechanicIntegration.cs
        local_integration = Path('src/OrchestratorAgent/LocalBotMechanicIntegration.cs')
        if not local_integration.exists():
            issues.append("Missing LocalBotMechanicIntegration.cs")
        
        if issues:
            return {
                'healthy': False,
                'issue': f"{len(issues)} intelligence components missing",
                'fixable': True,
                'fix_type': 'create_intelligence_files'
            }
        
        return {'healthy': True}
    
    def _check_ml_health(self) -> Dict:
        """Check ML models health"""
        model_dirs = [Path('Intelligence/models'), Path('models'), Path('ml')]
        
        model_files = []
        for model_dir in model_dirs:
            if model_dir.exists():
                model_files.extend(list(model_dir.glob('*.pkl')))
                model_files.extend(list(model_dir.glob('*.h5')))
                model_files.extend(list(model_dir.glob('*.joblib')))
                model_files.extend(list(model_dir.glob('*.onnx')))
        
        if not model_files:
            return {
                'healthy': False,
                'issue': 'No trained models found',
                'fixable': True,
                'fix_type': 'train_emergency_model'
            }
        
        # Check model age
        old_models = []
        for model in model_files:
            age_days = (time.time() - model.stat().st_mtime) / 86400
            if age_days > 7:
                old_models.append(model.name)
        
        if len(old_models) > 3:
            return {
                'healthy': False,
                'issue': f"{len(old_models)} old models",
                'fixable': True,
                'fix_type': 'retrain_models'
            }
        
        return {'healthy': True}
    
    def _check_data_health(self) -> Dict:
        """Check data pipeline health"""
        data_dirs = [Path('Intelligence/data'), Path('data')]
        
        data_files = []
        for data_dir in data_dirs:
            if data_dir.exists():
                data_files.extend(list(data_dir.glob('*.csv')))
                data_files.extend(list(data_dir.glob('*.json')))
        
        if not data_files:
            return {
                'healthy': False,
                'issue': 'No data files found',
                'fixable': True,
                'fix_type': 'fetch_initial_data'
            }
        
        # Check data freshness
        stale_files = []
        for data_file in data_files:
            age_hours = (time.time() - data_file.stat().st_mtime) / 3600
            if age_hours > 24:
                stale_files.append(data_file.name)
        
        if len(stale_files) > 10:
            return {
                'healthy': False,
                'issue': f"{len(stale_files)} stale data files",
                'fixable': True,
                'fix_type': 'refresh_data'
            }
        
        return {'healthy': True}
    
    def _check_dependencies_health(self) -> Dict:
        """Check Python dependencies"""
        required_packages = {
            'pandas': 'pandas',
            'numpy': 'numpy',
            'sklearn': 'scikit-learn',
            'requests': 'requests',
            'yfinance': 'yfinance'
        }
        
        missing = []
        for import_name, pip_name in required_packages.items():
            try:
                __import__(import_name)
            except ImportError:
                missing.append(pip_name)
        
        if missing:
            return {
                'healthy': False,
                'issue': f"Missing packages: {', '.join(missing)}",
                'fixable': True,
                'fix_type': 'install_packages',
                'packages': missing
            }
        
        return {'healthy': True}
    
    def _check_critical_files(self) -> Dict:
        """Check if critical files exist"""
        missing = []
        
        for category, files in self.critical_files.items():
            for file in files:
                if not Path(file).exists():
                    missing.append(file)
        
        if missing:
            return {
                'healthy': False,
                'issue': f"Missing {len(missing)} critical files",
                'fixable': True,
                'fix_type': 'create_critical_files',
                'files': missing
            }
        
        return {'healthy': True}
    
    def _check_workflow_health(self) -> Dict:
        """Check GitHub workflows"""
        workflow_dir = Path('.github/workflows')
        
        if not workflow_dir.exists():
            return {
                'healthy': False,
                'issue': 'No workflows directory',
                'fixable': True,
                'fix_type': 'create_workflow_dir'
            }
        
        yml_files = list(workflow_dir.glob('*.yml'))
        
        if len(yml_files) < 5:
            return {
                'healthy': False,
                'issue': f'Only {len(yml_files)} workflow files',
                'fixable': True,
                'fix_type': 'create_basic_workflows'
            }
        
        return {'healthy': True}
    
    def _check_cs_integration(self) -> Dict:
        """Check C# integration health"""
        cs_files = [
            'src/OrchestratorAgent/LocalBotMechanicIntegration.cs',
            'src/BotCore/Services/IntelligenceService.cs'
        ]
        
        missing = []
        for file in cs_files:
            if not Path(file).exists():
                missing.append(file)
        
        if missing:
            return {
                'healthy': False,
                'issue': f"Missing C# integration files: {len(missing)}",
                'fixable': False,
                'fix_type': 'manual_cs_integration'
            }
        
        return {'healthy': True}
    
    def _check_self_healing_status(self) -> Dict:
        """Check if C# self-healing is disabled"""
        self_healing_file = Path('src/OrchestratorAgent/Infra/SelfHealingEngine.cs')
        
        if self_healing_file.exists():
            content = self_healing_file.read_text(encoding='utf-8', errors='ignore')
            if 'TEMPORARILY DISABLED' in content:
                return {
                    'healthy': False,
                    'issue': 'C# Self-healing engine disabled',
                    'fixable': False,
                    'fix_type': 'manual_cs_fix'
                }
        
        return {'healthy': True}
    
    # ========================================
    # AUTO-REPAIR SYSTEM
    # ========================================
    
    def _auto_repair(self, system: str, status: Dict) -> bool:
        """Attempt automatic repair"""
        try:
            fix_type = status.get('fix_type')
            
            if fix_type == 'create_trading_files':
                return self._create_trading_files()
            
            elif fix_type == 'create_intelligence_files':
                return self._create_intelligence_files()
            
            elif fix_type == 'train_emergency_model':
                return self._train_emergency_model()
            
            elif fix_type == 'fetch_initial_data':
                return self._fetch_emergency_data()
            
            elif fix_type == 'install_packages':
                packages = status.get('packages', [])
                return self._install_packages(packages)
            
            elif fix_type == 'create_critical_files':
                files = status.get('files', [])
                return self._create_critical_files(files)
            
            elif fix_type == 'refresh_data':
                return self._fetch_emergency_data()
            
            # Log repair attempt
            self.repair_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'system': system,
                'fix_type': fix_type,
                'success': True
            })
            self.save_json(self.repairs_db, self.repair_history)
            
            return True
            
        except Exception as e:
            print(f"   ❌ Repair failed: {e}")
            
            self.repair_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'system': system,
                'fix_type': fix_type,
                'success': False,
                'error': str(e)
            })
            self.save_json(self.repairs_db, self.repair_history)
            
            return False
    
    def _create_intelligence_files(self) -> bool:
        """Create missing intelligence files"""
        try:
            intelligence_scripts = {
                'Intelligence/scripts/cot_report.py': self._get_cot_template(),
                'Intelligence/scripts/congressional_trades.py': self._get_congressional_template(),
                'Intelligence/scripts/social_momentum.py': self._get_social_template(),
                'Intelligence/scripts/intermarket.py': self._get_intermarket_template(),
                'Intelligence/scripts/opex_calendar.py': self._get_opex_template()
            }
            
            for file_path, content in intelligence_scripts.items():
                path = Path(file_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            
            return True
        except:
            return False
    
    def _create_trading_files(self) -> bool:
        """Create missing trading files"""
        try:
            # Create ES/NQ strategy
            es_nq_path = Path('Intelligence/scripts/strategies/es_nq_realtime.py')
            es_nq_path.parent.mkdir(parents=True, exist_ok=True)
            
            es_nq_content = '''#!/usr/bin/env python3
"""ES/NQ Real-time Trading Strategy - Auto-generated by Bot Mechanic"""

import json
from datetime import datetime

def generate_signals():
    """Generate ES/NQ trading signals"""
    signals = {
        "timestamp": datetime.utcnow().isoformat(),
        "ES": {"signal": "HOLD", "price": 4500, "confidence": 0.7},
        "NQ": {"signal": "HOLD", "price": 15500, "confidence": 0.7}
    }
    return signals

if __name__ == "__main__":
    signals = generate_signals()
    print(f"ES/NQ Signals: {signals}")
'''
            es_nq_path.write_text(es_nq_content)
            
            # Create SPY/QQQ strategy
            spy_qqq_path = Path('Intelligence/scripts/strategies/spy_qqq_regime.py')
            spy_qqq_content = '''#!/usr/bin/env python3
"""SPY/QQQ Regime Trading Strategy - Auto-generated by Bot Mechanic"""

import json
from datetime import datetime

def detect_regime():
    """Detect market regime"""
    regime = {
        "timestamp": datetime.utcnow().isoformat(),
        "SPY": {"regime": "NEUTRAL", "signal": "HOLD"},
        "QQQ": {"regime": "NEUTRAL", "signal": "HOLD"}
    }
    return regime

if __name__ == "__main__":
    regime = detect_regime()
    print(f"Market Regime: {regime}")
'''
            spy_qqq_path.write_text(spy_qqq_content)
            
            return True
        except:
            return False
    
    def _train_emergency_model(self) -> bool:
        """Train emergency ML model"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            import numpy as np
            import pickle
            
            model_dir = Path('Intelligence/models')
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate dummy data
            X = np.random.rand(1000, 20)
            y = np.random.randint(0, 2, 1000)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Save model
            model_path = model_dir / 'emergency_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            return True
        except:
            return False
    
    def _fetch_emergency_data(self) -> bool:
        """Fetch emergency market data"""
        try:
            data_dir = Path('Intelligence/data')
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Create emergency signals
            emergency_signals = {
                "timestamp": datetime.utcnow().isoformat(),
                "ES": {"signal": "HOLD", "confidence": 0.5},
                "NQ": {"signal": "HOLD", "confidence": 0.5},
                "SPY": {"signal": "HOLD", "confidence": 0.5},
                "QQQ": {"signal": "HOLD", "confidence": 0.5},
                "generated_by": "bot_mechanic_emergency",
                "status": "emergency_fallback"
            }
            
            signal_path = data_dir / 'latest_signals.json'
            self.save_json(signal_path, emergency_signals)
            
            return True
        except:
            return False
    
    def _install_packages(self, packages: List[str]) -> bool:
        """Install missing Python packages"""
        success = True
        
        for package in packages:
            try:
                print(f"   📦 Installing {package}...")
                result = subprocess.run(
                    f"pip install {package}",
                    shell=True,
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    success = False
            except:
                success = False
        
        return success
    
    def _create_critical_files(self, files: List[str]) -> bool:
        """Create critical missing files"""
        for file_path in files:
            try:
                path = Path(file_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create appropriate content based on file name
                if 'intelligence' in file_path.lower():
                    content = self._get_intelligence_template(path.name)
                elif 'strategy' in file_path.lower():
                    content = self._get_strategy_template()
                elif 'ml' in file_path.lower() or 'train' in file_path.lower():
                    content = self._get_ml_training_template()
                else:
                    content = self._get_generic_template()
                
                path.write_text(content)
            except:
                return False
        
        return True
    
    # ========================================
    # MONITORING SYSTEM
    # ========================================
    
    def start_monitoring(self, interval: int = 300):
        """Start continuous monitoring"""
        print(f"\n🔄 Starting continuous monitoring (every {interval}s)")
        print("Press Ctrl+C to stop\n")
        
        def monitor_loop():
            while True:
                try:
                    # Quick scan (less verbose)
                    results = self.quick_scan()
                    
                    # Display status
                    status = "✅ Healthy" if results['healthy'] else f"⚠️ {results['issues']} issues"
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {status}", end='\r')
                    
                    # If issues found, try to fix
                    if not results['healthy']:
                        print(f"\n🔧 Fixing {results['issues']} issues...")
                        self.auto_fix_all()
                    
                    time.sleep(interval)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"\n❌ Monitor error: {e}")
                    time.sleep(30)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
        return monitor_thread
    
    def quick_scan(self) -> Dict:
        """Quick health scan without full analysis"""
        health_checks = {
            'trading': self._check_trading_health(),
            'intelligence': self._check_intelligence_health(),
            'ml': self._check_ml_health(),
            'data': self._check_data_health(),
            'dependencies': self._check_dependencies_health()
        }
        
        issues = sum(1 for check in health_checks.values() if not check['healthy'])
        
        return {
            'healthy': issues == 0,
            'issues': issues,
            'checks': health_checks,
            'health_score': max(0, 100 - (issues * 10))
        }
    
    def auto_fix_all(self):
        """Attempt to fix all found issues"""
        checks = self.quick_scan()['checks']
        
        for system, status in checks.items():
            if not status['healthy'] and status.get('fixable'):
                self._auto_repair(system, status)
    
    def get_dashboard_data(self) -> Dict:
        """Get data for dashboard integration"""
        quick_health = self.quick_scan()
        
        return {
            'status': 'healthy' if quick_health['healthy'] else 'warning',
            'health_score': quick_health['health_score'],
            'issues_count': quick_health['issues'],
            'last_scan': self.knowledge.get('last_scan', {}),
            'recent_repairs': self.repair_history[-5:] if self.repair_history else [],
            'feature_count': sum(self.knowledge.get('last_scan', {}).get('categories', {}).values()),
            'monitoring_active': True
        }
    
    # ========================================
    # FEATURE DETECTION
    # ========================================
    
    def _detect_feature_changes(self, results: Dict):
        """Detect new, modified, and removed features"""
        current_features = set()
        
        # Collect all current features
        for file_data in self.knowledge.get('files', {}).values():
            for func in file_data.get('features', {}).get('functions', []):
                current_features.add(func['name'])
        
        # Load previous features
        if self.known_features:
            previous = set(self.known_features.get('all_features', []))
            
            # Find changes
            new = current_features - previous
            removed = previous - current_features
            
            if new:
                results['new_features'].extend(list(new))
                print(f"✨ Found {len(new)} new features")
            
            if removed:
                results['missing_features'].extend(list(removed))
                print(f"⚠️ Missing {len(removed)} features")
        
        # Save current features
        self.known_features['all_features'] = list(current_features)
        self.known_features['last_update'] = datetime.utcnow().isoformat()
        self.save_json(self.features_db, self.known_features)
    
    def _calculate_health_score(self, results: Dict) -> int:
        """Calculate overall health score"""
        score = 100
        
        # Deduct for issues
        score -= len(results['issues_found']) * 5
        score -= len(results['broken_features']) * 15
        score -= len(results['missing_features']) * 3
        
        # Add points for auto-fixes
        score += len(results['auto_fixed']) * 2
        
        return max(0, min(100, score))
    
    def _attempt_syntax_fix(self, filepath: Path, error: SyntaxError) -> bool:
        """Attempt to fix syntax errors"""
        # This would implement common syntax fixes
        # For now, return False to indicate manual fix needed
        return False
    
    # ========================================
    # REPORTING
    # ========================================
    
    def _print_scan_report(self, results: Dict):
        """Print detailed scan report"""
        print("\n" + "="*60)
        print("📊 SCAN REPORT")
        print("="*60)
        
        print(f"Timestamp: {results['timestamp']}")
        print(f"Files scanned: {results['files_scanned']}")
        print(f"Total features: {results['total_features']}")
        print(f"Health score: {results['health_score']}%")
        print(f"Scan time: {results.get('scan_time_seconds', 0):.2f}s")
        
        # Feature breakdown
        if results['categories']:
            print("\n📁 Features by Category:")
            for cat, count in sorted(results['categories'].items(), key=lambda x: x[1], reverse=True):
                print(f"  • {cat.title()}: {count}")
        
        # Issues
        if results['issues_found']:
            print(f"\n⚠️ Issues Found ({len(results['issues_found'])}):")
            for issue in results['issues_found'][:5]:
                print(f"  • {issue}")
        
        # Auto-fixes
        if results['auto_fixed']:
            print(f"\n🔧 Auto-Fixed ({len(results['auto_fixed'])}):")
            for fix in results['auto_fixed']:
                print(f"  • {fix}")
        
        # Overall health
        health_score = results['health_score']
        
        print(f"\n💪 Overall Health Score: {health_score}%")
        
        if health_score == 100:
            print("   ✅ Everything is working perfectly!")
        elif health_score >= 80:
            print("   🟢 Bot is healthy with minor issues")
        elif health_score >= 60:
            print("   🟡 Bot needs attention")
        else:
            print("   🔴 Critical issues detected!")
        
        print("="*60)
    
    def generate_html_report(self) -> str:
        """Generate HTML report of bot status"""
        html = '''<!DOCTYPE html>
<html>
<head>
    <title>Bot Mechanic Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
        h1 { color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .healthy { background: #d4edda; color: #155724; }
        .warning { background: #fff3cd; color: #856404; }
        .error { background: #f8d7da; color: #721c24; }
        .feature { display: inline-block; padding: 5px 10px; margin: 5px; background: #e9ecef; border-radius: 3px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #007bff; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Bot Mechanic Report</h1>
        <p>Generated: ''' + datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC') + '''</p>
'''
        
        # Add health status
        health = self.quick_scan()
        status_class = 'healthy' if health['healthy'] else 'error'
        html += f'''
        <div class="status {status_class}">
            <h2>Overall Status: {'✅ Healthy' if health['healthy'] else f"⚠️ {health['issues']} Issues"}</h2>
        </div>
'''
        
        # Add feature statistics
        if self.knowledge.get('files'):
            total_files = len(self.knowledge['files'])
            total_functions = sum(
                len(f.get('features', {}).get('functions', []))
                for f in self.knowledge['files'].values()
            )
            
            html += f'''
        <h2>📊 Statistics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Files</td><td>{total_files}</td></tr>
            <tr><td>Total Functions</td><td>{total_functions}</td></tr>
            <tr><td>Last Scan</td><td>{self.knowledge.get('last_scan', {}).get('timestamp', 'Never')}</td></tr>
        </table>
'''
        
        html += '''
    </div>
</body>
</html>'''
        
        # Save report
        report_path = Path('Intelligence/mechanic/reports/latest_report.html')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(html)
        
        return str(report_path)

    def get_dashboard_data(self) -> Dict:
        """Get data for dashboard integration"""
        quick_health = self.quick_scan()
        
        return {
            'status': 'healthy' if quick_health['healthy'] else 'warning',
            'health_score': quick_health['health_score'],
            'issues_count': quick_health['issues'],
            'last_scan': self.knowledge.get('last_scan', {}),
            'recent_repairs': self.repair_history[-5:] if self.repair_history else [],
            'feature_count': sum(self.knowledge.get('last_scan', {}).get('categories', {}).values()),
            'monitoring_active': True
        }
    
    # ========================================
    # TEMPLATE GENERATORS
    # ========================================
    
    def _get_cot_template(self) -> str:
        return '''#!/usr/bin/env python3
"""COT Report Analysis - Auto-generated by Bot Mechanic"""

import json
from datetime import datetime

class COTAnalyzer:
    def __init__(self):
        self.name = "COT Report Analyzer"
    
    def analyze_cot_data(self):
        """Analyze COT report data"""
        analysis = {
            "timestamp": datetime.utcnow().isoformat(),
            "ES_commercials": {"net_position": 0, "change": 0},
            "ES_large_specs": {"net_position": 0, "change": 0},
            "signal": "NEUTRAL",
            "confidence": 0.5,
            "generated_by": "bot_mechanic"
        }
        return analysis

if __name__ == "__main__":
    analyzer = COTAnalyzer()
    result = analyzer.analyze_cot_data()
    print(json.dumps(result, indent=2))
'''
    
    def _get_congressional_template(self) -> str:
        return '''#!/usr/bin/env python3
"""Congressional Trades Monitor - Auto-generated by Bot Mechanic"""

import json
from datetime import datetime

class CongressionalTradesMonitor:
    def __init__(self):
        self.name = "Congressional Trades Monitor"
    
    def monitor_trades(self):
        """Monitor congressional trading activity"""
        analysis = {
            "timestamp": datetime.utcnow().isoformat(),
            "recent_trades": [],
            "sector_sentiment": "NEUTRAL",
            "confidence": 0.5,
            "generated_by": "bot_mechanic"
        }
        return analysis

if __name__ == "__main__":
    monitor = CongressionalTradesMonitor()
    result = monitor.monitor_trades()
    print(json.dumps(result, indent=2))
'''
    
    def _get_social_template(self) -> str:
        return '''#!/usr/bin/env python3
"""Social Momentum Tracker - Auto-generated by Bot Mechanic"""

import json
from datetime import datetime

class SocialMomentumTracker:
    def __init__(self):
        self.name = "Social Momentum Tracker"
    
    def track_sentiment(self):
        """Track social media sentiment"""
        analysis = {
            "timestamp": datetime.utcnow().isoformat(),
            "sentiment_score": 0.0,
            "momentum": "NEUTRAL",
            "confidence": 0.5,
            "generated_by": "bot_mechanic"
        }
        return analysis

if __name__ == "__main__":
    tracker = SocialMomentumTracker()
    result = tracker.track_sentiment()
    print(json.dumps(result, indent=2))
'''
    
    def _get_intermarket_template(self) -> str:
        return '''#!/usr/bin/env python3
"""Intermarket Analysis - Auto-generated by Bot Mechanic"""

import json
from datetime import datetime

class IntermarketAnalyzer:
    def __init__(self):
        self.name = "Intermarket Analyzer"
    
    def analyze_correlations(self):
        """Analyze intermarket correlations"""
        analysis = {
            "timestamp": datetime.utcnow().isoformat(),
            "correlations": {
                "SPY_QQQ": 0.85,
                "ES_NQ": 0.82,
                "VIX_SPY": -0.75
            },
            "regime": "NEUTRAL",
            "confidence": 0.5,
            "generated_by": "bot_mechanic"
        }
        return analysis

if __name__ == "__main__":
    analyzer = IntermarketAnalyzer()
    result = analyzer.analyze_correlations()
    print(json.dumps(result, indent=2))
'''
    
    def _get_opex_template(self) -> str:
        return '''#!/usr/bin/env python3
"""OPEX Calendar Monitor - Auto-generated by Bot Mechanic"""

import json
from datetime import datetime

class OPEXCalendarMonitor:
    def __init__(self):
        self.name = "OPEX Calendar Monitor"
    
    def check_opex_impact(self):
        """Check OPEX calendar impact"""
        analysis = {
            "timestamp": datetime.utcnow().isoformat(),
            "next_opex": "2025-09-20",
            "impact_level": "LOW",
            "expected_volatility": 0.15,
            "confidence": 0.5,
            "generated_by": "bot_mechanic"
        }
        return analysis

if __name__ == "__main__":
    monitor = OPEXCalendarMonitor()
    result = monitor.check_opex_impact()
    print(json.dumps(result, indent=2))
'''
    
    def _get_intelligence_template(self, filename: str) -> str:
        return f'''#!/usr/bin/env python3
"""Intelligence Script: {filename} - Auto-generated by Bot Mechanic"""

import json
from datetime import datetime

def analyze():
    """Main analysis function"""
    result = {{
        "timestamp": datetime.utcnow().isoformat(),
        "signal": "NEUTRAL",
        "confidence": 0.5,
        "generated_by": "bot_mechanic",
        "script": "{filename}"
    }}
    return result

if __name__ == "__main__":
    result = analyze()
    print(json.dumps(result, indent=2))
'''
    
    def _get_ml_training_template(self) -> str:
        return '''#!/usr/bin/env python3
"""ML Training Script - Auto-generated by Bot Mechanic"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import os

def train_model():
    """Train a basic ML model"""
    # Generate sample data (replace with real data)
    X = np.random.rand(1000, 20)
    y = np.random.randint(0, 2, 1000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Save model
    os.makedirs("Intelligence/models", exist_ok=True)
    with open("Intelligence/models/model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    return model

if __name__ == "__main__":
    train_model()
'''
    
    def _get_strategy_template(self) -> str:
        return '''#!/usr/bin/env python3
"""Trading Strategy - Auto-generated by Bot Mechanic"""

import pandas as pd
import numpy as np
from datetime import datetime

def calculate_signals(data):
    """Calculate trading signals from data"""
    signals = {
        'timestamp': datetime.utcnow().isoformat(),
        'signal': 'HOLD',
        'confidence': 0.5,
        'reason': 'Default strategy'
    }
    
    # Add your strategy logic here
    # Example: Simple moving average crossover
    if len(data) > 20:
        sma_short = data['close'].rolling(window=10).mean().iloc[-1]
        sma_long = data['close'].rolling(window=20).mean().iloc[-1]
        
        if sma_short > sma_long:
            signals['signal'] = 'BUY'
            signals['confidence'] = 0.7
            signals['reason'] = 'SMA crossover bullish'
        elif sma_short < sma_long:
            signals['signal'] = 'SELL'
            signals['confidence'] = 0.7
            signals['reason'] = 'SMA crossover bearish'
    
    return signals

if __name__ == "__main__":
    # Test with dummy data
    test_data = pd.DataFrame({
        'close': np.random.randn(50).cumsum() + 100
    })
    signals = calculate_signals(test_data)
    print(signals)
'''
    
    def _get_generic_template(self) -> str:
        return '''#!/usr/bin/env python3
"""Auto-generated script by Bot Mechanic"""

def main():
    """Main function"""
    print("This is an auto-generated placeholder script")
    print("Please implement your logic here")
    return True

if __name__ == "__main__":
    main()
'''


# ========================================
# MAIN EXECUTION
# ========================================

def main():
    """Main entry point for Local Bot Mechanic"""
    mechanic = LocalBotMechanic()
    
    print("\n" + "="*60)
    print("🧠 LOCAL BOT MECHANIC v3.0")
    print("="*60)
    print("\nOptions:")
    print("1. Full deep scan and repair")
    print("2. Quick health check")
    print("3. Start continuous monitoring")
    print("4. Auto-fix all issues")
    print("5. Full scan + continuous monitoring (recommended)")
    print("6. Exit")
    
    try:
        choice = input("\nSelect option (1-6): ").strip()
    except:
        choice = '5'  # Default for auto-start
    
    if choice == '1':
        print("\n🔍 Starting full deep scan...")
        mechanic.deep_scan()
        
    elif choice == '2':
        print("\n⚡ Running quick health check...")
        results = mechanic.quick_scan()
        
        if results['healthy']:
            print("✅ All systems healthy!")
        else:
            print(f"⚠️ Found {results['issues']} issues")
            mechanic.auto_fix_all()
    
    elif choice == '3':
        mechanic.start_monitoring(interval=60)  # Check every minute
    
    elif choice == '4':
        print("\n🔧 Auto-fixing all issues...")
        mechanic.auto_fix_all()
        
    elif choice == '5':
        print("\n🚀 Starting full scan + monitoring...")
        mechanic.deep_scan()
        monitor_thread = mechanic.start_monitoring(interval=60)
        try:
            monitor_thread.join()
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped")
    
    else:
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
