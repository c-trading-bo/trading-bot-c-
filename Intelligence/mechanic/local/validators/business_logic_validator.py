#!/usr/bin/env python3
"""
BUSINESS LOGIC VALIDATOR - Complete System
Validates all trading logic, risk management, algorithms, and semantic correctness
"""

import os
import ast
import json
import re
import inspect
import importlib
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

@dataclass
class ValidationResult:
    """Validation result structure"""
    passed: bool
    category: str
    rule: str
    details: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    file: str = ""
    line: int = 0
    suggestion: str = ""

class BusinessLogicValidator:
    def __init__(self, mechanic_instance=None):
        self.version = "4.0-BLV"
        self.mechanic = mechanic_instance
        self.base_path = Path.cwd()
        
        # Validation databases
        self.validation_db = Path("Intelligence/mechanic/database/validation/validation_results.json")
        self.rules_db = Path("Intelligence/mechanic/database/validation/rules.json")
        self.patterns_db = Path("Intelligence/mechanic/database/validation/patterns.json")
        
        # Create directories
        self.validation_db.parent.mkdir(parents=True, exist_ok=True)
        
        # Load validation rules
        self.rules = self.load_validation_rules()
        self.patterns = self.load_patterns()
        
        # Validation results
        self.results = []
        self.critical_issues = []
        
        # Trading constants for validation
        self.TRADING_RULES = {
            'max_position_size': 1000000,  # $1M max
            'max_leverage': 10,
            'max_risk_per_trade': 0.02,  # 2% max risk
            'min_risk_reward': 1.5,  # Minimum 1.5:1 RR
            'max_daily_loss': 0.05,  # 5% daily loss limit
            'min_stop_loss': 0.001,  # 0.1% minimum SL
            'max_stop_loss': 0.10,  # 10% maximum SL
            'max_positions': 10,  # Max concurrent positions
            'min_liquidity': 1000000,  # $1M daily volume minimum
        }
        
        # Algorithm validation thresholds
        self.ALGO_RULES = {
            'min_sharpe_ratio': 0.5,
            'max_drawdown': 0.20,  # 20% max drawdown
            'min_win_rate': 0.40,  # 40% minimum win rate
            'min_profit_factor': 1.2,
            'max_consecutive_losses': 10,
            'min_backtest_days': 30,
            'min_sample_size': 100,
        }
        
        print(f"🧠 Business Logic Validator v{self.version} initialized")
    
    def load_validation_rules(self) -> Dict:
        """Load validation rules"""
        default_rules = {
            'risk_management': [
                {'rule': 'position_sizing', 'check': 'validate_position_sizing'},
                {'rule': 'stop_loss_required', 'check': 'validate_stop_loss'},
                {'rule': 'risk_reward_ratio', 'check': 'validate_risk_reward'},
                {'rule': 'max_exposure', 'check': 'validate_max_exposure'},
                {'rule': 'leverage_limits', 'check': 'validate_leverage'},
            ],
            'trading_logic': [
                {'rule': 'entry_conditions', 'check': 'validate_entry_logic'},
                {'rule': 'exit_conditions', 'check': 'validate_exit_logic'},
                {'rule': 'order_validation', 'check': 'validate_order_logic'},
                {'rule': 'signal_generation', 'check': 'validate_signal_logic'},
                {'rule': 'timing_rules', 'check': 'validate_timing_logic'},
            ],
            'algorithms': [
                {'rule': 'formula_correctness', 'check': 'validate_formulas'},
                {'rule': 'indicator_calculation', 'check': 'validate_indicators'},
                {'rule': 'backtest_integrity', 'check': 'validate_backtest'},
                {'rule': 'performance_metrics', 'check': 'validate_metrics'},
                {'rule': 'numerical_stability', 'check': 'validate_numerical'},
            ],
            'semantic': [
                {'rule': 'data_flow', 'check': 'validate_data_flow'},
                {'rule': 'variable_usage', 'check': 'validate_variables'},
                {'rule': 'dependency_chain', 'check': 'validate_dependencies'},
                {'rule': 'logic_consistency', 'check': 'validate_logic_flow'},
                {'rule': 'error_handling', 'check': 'validate_error_handling'},
            ]
        }
        
        # Load custom rules if exist
        if self.rules_db.exists():
            try:
                with open(self.rules_db) as f:
                    custom_rules = json.load(f)
                    default_rules.update(custom_rules)
            except:
                pass
        
        return default_rules
    
    def load_patterns(self) -> Dict:
        """Load validation patterns"""
        return {
            'dangerous_patterns': [
                r'exec\s*\(',  # Dynamic execution
                r'eval\s*\(',  # Eval usage
                r'__import__',  # Dynamic imports
                r'os\.system',  # System calls
                r'subprocess\.',  # Subprocess usage
            ],
            'risk_patterns': [
                r'position_size\s*=\s*capital\s*\*\s*(\d+\.?\d*)',  # Position sizing
                r'stop_loss\s*=\s*.*?(\d+\.?\d*)',  # Stop loss
                r'leverage\s*=\s*(\d+)',  # Leverage usage
                r'risk\s*=\s*(\d+\.?\d*)',  # Risk percentage
            ],
            'trading_patterns': [
                r'if\s+.*?buy.*?:',  # Buy conditions
                r'if\s+.*?sell.*?:',  # Sell conditions
                r'signal\s*=\s*["\']?(BUY|SELL|HOLD)',  # Signal generation
                r'order\s*=.*?place_order',  # Order placement
            ],
            'algorithm_patterns': [
                r'sma\s*=.*?rolling.*?mean',  # SMA calculation
                r'rsi\s*=.*?',  # RSI calculation
                r'macd\s*=.*?',  # MACD calculation
                r'backtest.*?\(',  # Backtest function
            ]
        }
    
    # ========================================
    # MAIN VALIDATION ENGINE
    # ========================================
    
    def validate_entire_bot(self, verbose: bool = True) -> Dict:
        """Complete validation of entire bot"""
        if verbose:
            print("\n" + "="*60)
            print("🔍 BUSINESS LOGIC VALIDATION - STARTING")
            print("="*60)
        
        validation_start = datetime.utcnow()
        
        # Clear previous results
        self.results = []
        self.critical_issues = []
        
        # Run all validations
        self._validate_risk_management(verbose)
        self._validate_trading_logic(verbose)
        self._validate_algorithms(verbose)
        self._validate_semantic_analysis(verbose)
        
        # Analyze results
        summary = self._analyze_results()
        
        # Save results
        self._save_validation_results(summary)
        
        # Print report
        if verbose:
            self._print_validation_report(summary)
        
        return summary
    
    # ========================================
    # RISK MANAGEMENT VALIDATION
    # ========================================
    
    def _validate_risk_management(self, verbose: bool):
        """Validate all risk management logic"""
        if verbose:
            print("\n💰 VALIDATING RISK MANAGEMENT...")
        
        # Find all files with risk management code
        risk_files = self._find_files_with_pattern(['risk', 'position', 'stop', 'size'])
        
        for file_path in risk_files:
            self._validate_risk_file(file_path)
    
    def _validate_risk_file(self, file_path: Path):
        """Validate risk management in a file"""
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check position sizing functions
                    if 'position' in node.name.lower() or 'size' in node.name.lower():
                        self._validate_position_sizing_logic(node, file_path, content)
                    
                    # Check stop loss functions
                    if 'stop' in node.name.lower() or 'sl' in node.name.lower():
                        self._validate_stop_loss_logic(node, file_path, content)
                    
                    # Check risk calculation functions
                    if 'risk' in node.name.lower():
                        self._validate_risk_calculation(node, file_path, content)
        
        except Exception as e:
            self.results.append(ValidationResult(
                passed=False,
                category='risk_management',
                rule='file_parse_error',
                details=f"Failed to parse {file_path}: {e}",
                severity='HIGH',
                file=str(file_path)
            ))
    
    def _validate_position_sizing_logic(self, node: ast.FunctionDef, file_path: Path, content: str):
        """Validate position sizing logic"""
        source_lines = content.splitlines()
        
        # Check for maximum position size
        has_max_check = False
        has_capital_check = False
        
        for child in ast.walk(node):
            # Look for comparisons
            if isinstance(child, ast.Compare):
                try:
                    compare_str = ast.unparse(child) if hasattr(ast, 'unparse') else str(child)
                    if 'position' in compare_str.lower():
                        has_max_check = True
                except:
                    pass
            
            # Look for capital/balance reference
            if isinstance(child, ast.Name):
                if child.id in ['capital', 'balance', 'equity', 'account_size']:
                    has_capital_check = True
        
        if not has_max_check:
            self.results.append(ValidationResult(
                passed=False,
                category='risk_management',
                rule='position_sizing_limit',
                details=f"Function '{node.name}' lacks maximum position size check",
                severity='CRITICAL',
                file=str(file_path),
                line=node.lineno,
                suggestion="Add: if position_size > max_position_size: position_size = max_position_size"
            ))
            self.critical_issues.append(f"Missing position limit in {node.name}")
        
        if not has_capital_check:
            self.results.append(ValidationResult(
                passed=False,
                category='risk_management',
                rule='position_sizing_capital',
                details=f"Function '{node.name}' doesn't reference account capital",
                severity='HIGH',
                file=str(file_path),
                line=node.lineno,
                suggestion="Position size should be based on account capital"
            ))
    
    def _validate_stop_loss_logic(self, node: ast.FunctionDef, file_path: Path, content: str):
        """Validate stop loss logic"""
        has_stop_loss = False
        stop_loss_values = []
        
        for child in ast.walk(node):
            # Look for stop loss assignments
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        if 'stop' in target.id.lower() or 'sl' in target.id.lower():
                            has_stop_loss = True
                            
                            # Extract value if it's a constant
                            if isinstance(child.value, ast.Constant):
                                stop_loss_values.append(child.value.value)
        
        if not has_stop_loss:
            self.results.append(ValidationResult(
                passed=False,
                category='risk_management',
                rule='missing_stop_loss',
                details=f"Function '{node.name}' doesn't set stop loss",
                severity='CRITICAL',
                file=str(file_path),
                line=node.lineno,
                suggestion="Always set a stop loss for every position"
            ))
            self.critical_issues.append(f"No stop loss in {node.name}")
    
    def _validate_risk_calculation(self, node: ast.FunctionDef, file_path: Path, content: str):
        """Validate risk calculation logic"""
        # Check for proper risk calculation formula
        has_division = False
        has_capital_reference = False
        
        for child in ast.walk(node):
            if isinstance(child, ast.BinOp) and isinstance(child.op, ast.Div):
                has_division = True
            
            if isinstance(child, ast.Name):
                if child.id in ['capital', 'balance', 'equity']:
                    has_capital_reference = True
        
        if not (has_division and has_capital_reference):
            self.results.append(ValidationResult(
                passed=False,
                category='risk_management',
                rule='risk_calculation',
                details=f"Function '{node.name}' may have incorrect risk calculation",
                severity='HIGH',
                file=str(file_path),
                line=node.lineno,
                suggestion="Risk should be calculated as: (position_size * stop_loss) / capital"
            ))
    
    # ========================================
    # TRADING LOGIC VALIDATION
    # ========================================
    
    def _validate_trading_logic(self, verbose: bool):
        """Validate all trading logic"""
        if verbose:
            print("\n📈 VALIDATING TRADING LOGIC...")
        
        # Find trading strategy files
        trading_files = self._find_files_with_pattern(['trade', 'strategy', 'signal', 'order'])
        
        for file_path in trading_files:
            self._validate_trading_file(file_path)
    
    def _validate_trading_file(self, file_path: Path):
        """Validate trading logic in a file"""
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check signal generation
                    if 'signal' in node.name.lower() or 'generate' in node.name.lower():
                        self._validate_signal_generation(node, file_path, content)
                    
                    # Check entry logic
                    if 'entry' in node.name.lower() or 'enter' in node.name.lower():
                        self._validate_entry_logic(node, file_path, content)
                    
                    # Check exit logic
                    if 'exit' in node.name.lower() or 'close' in node.name.lower():
                        self._validate_exit_logic(node, file_path, content)
                    
                    # Check order logic
                    if 'order' in node.name.lower():
                        self._validate_order_logic(node, file_path, content)
        
        except Exception as e:
            self.results.append(ValidationResult(
                passed=False,
                category='trading_logic',
                rule='file_parse_error',
                details=f"Failed to parse {file_path}: {e}",
                severity='HIGH',
                file=str(file_path)
            ))
    
    def _validate_signal_generation(self, node: ast.FunctionDef, file_path: Path, content: str):
        """Validate signal generation logic"""
        valid_signals = ['BUY', 'SELL', 'HOLD', 'LONG', 'SHORT', 'NEUTRAL']
        has_valid_signal = False
        has_conditions = False
        
        for child in ast.walk(node):
            # Check for valid signal values
            if isinstance(child, ast.Constant):
                if isinstance(child.value, str) and child.value.upper() in valid_signals:
                    has_valid_signal = True
            
            # Check for conditional logic
            if isinstance(child, (ast.If, ast.Compare)):
                has_conditions = True
        
        if not has_valid_signal:
            self.results.append(ValidationResult(
                passed=False,
                category='trading_logic',
                rule='invalid_signals',
                details=f"Function '{node.name}' doesn't return valid signals {valid_signals}",
                severity='HIGH',
                file=str(file_path),
                line=node.lineno,
                suggestion=f"Return one of: {', '.join(valid_signals)}"
            ))
        
        if not has_conditions:
            self.results.append(ValidationResult(
                passed=False,
                category='trading_logic',
                rule='no_signal_conditions',
                details=f"Function '{node.name}' lacks conditional logic for signals",
                severity='CRITICAL',
                file=str(file_path),
                line=node.lineno,
                suggestion="Add conditional logic (if/else) for signal generation"
            ))
            self.critical_issues.append(f"No conditions in {node.name}")
    
    def _validate_entry_logic(self, node: ast.FunctionDef, file_path: Path, content: str):
        """Validate entry logic"""
        checks = {
            'has_price_check': False,
            'has_volume_check': False,
            'has_indicator_check': False,
            'has_risk_check': False
        }
        
        for child in ast.walk(node):
            if isinstance(child, ast.Compare):
                try:
                    compare_str = ast.unparse(child) if hasattr(ast, 'unparse') else ''
                    
                    if 'price' in compare_str.lower():
                        checks['has_price_check'] = True
                    if 'volume' in compare_str.lower():
                        checks['has_volume_check'] = True
                    if any(ind in compare_str.lower() for ind in ['rsi', 'macd', 'sma', 'ema']):
                        checks['has_indicator_check'] = True
                    if 'risk' in compare_str.lower():
                        checks['has_risk_check'] = True
                except:
                    pass
        
        # Require at least price and one other check
        if not checks['has_price_check']:
            self.results.append(ValidationResult(
                passed=False,
                category='trading_logic',
                rule='entry_price_check',
                details=f"Entry function '{node.name}' doesn't check price",
                severity='HIGH',
                file=str(file_path),
                line=node.lineno,
                suggestion="Add price validation before entry"
            ))
    
    def _validate_exit_logic(self, node: ast.FunctionDef, file_path: Path, content: str):
        """Validate exit logic"""
        exit_conditions = {
            'profit_target': False,
            'stop_loss': False,
            'time_exit': False,
            'signal_exit': False
        }
        
        try:
            source = ast.unparse(node) if hasattr(ast, 'unparse') else content
        except:
            source = content
        
        # Check for exit conditions
        if 'profit' in source.lower() or 'target' in source.lower():
            exit_conditions['profit_target'] = True
        if 'stop' in source.lower() or 'loss' in source.lower():
            exit_conditions['stop_loss'] = True
        if 'time' in source.lower() or 'duration' in source.lower():
            exit_conditions['time_exit'] = True
        if 'signal' in source.lower():
            exit_conditions['signal_exit'] = True
        
        # Must have at least stop loss
        if not exit_conditions['stop_loss']:
            self.results.append(ValidationResult(
                passed=False,
                category='trading_logic',
                rule='exit_stop_loss',
                details=f"Exit function '{node.name}' doesn't implement stop loss",
                severity='CRITICAL',
                file=str(file_path),
                line=node.lineno,
                suggestion="Always implement stop loss in exit logic"
            ))
            self.critical_issues.append(f"No stop loss exit in {node.name}")
    
    def _validate_order_logic(self, node: ast.FunctionDef, file_path: Path, content: str):
        """Validate order placement logic"""
        order_checks = {
            'has_validation': False,
            'has_size_check': False,
            'has_error_handling': False,
            'has_confirmation': False
        }
        
        for child in ast.walk(node):
            # Check for validation
            if isinstance(child, ast.If):
                order_checks['has_validation'] = True
            
            # Check for size validation
            if isinstance(child, ast.Compare):
                try:
                    compare_str = ast.unparse(child) if hasattr(ast, 'unparse') else ''
                    if 'size' in compare_str.lower() or 'quantity' in compare_str.lower():
                        order_checks['has_size_check'] = True
                except:
                    pass
            
            # Check for error handling
            if isinstance(child, ast.Try):
                order_checks['has_error_handling'] = True
        
        if not order_checks['has_error_handling']:
            self.results.append(ValidationResult(
                passed=False,
                category='trading_logic',
                rule='order_error_handling',
                details=f"Order function '{node.name}' lacks error handling",
                severity='CRITICAL',
                file=str(file_path),
                line=node.lineno,
                suggestion="Add try-except for order submission"
            ))
            self.critical_issues.append(f"No error handling in order: {node.name}")
    
    # ========================================
    # ALGORITHM VALIDATION
    # ========================================
    
    def _validate_algorithms(self, verbose: bool):
        """Validate all algorithms and calculations"""
        if verbose:
            print("\n🔬 VALIDATING ALGORITHMS...")
        
        # Find algorithm files
        algo_files = self._find_files_with_pattern(['algorithm', 'calculate', 'indicator', 'backtest', 'model'])
        
        for file_path in algo_files:
            self._validate_algorithm_file(file_path)
    
    def _validate_algorithm_file(self, file_path: Path):
        """Validate algorithms in a file"""
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check indicator calculations
                    if any(ind in node.name.lower() for ind in ['rsi', 'macd', 'sma', 'ema', 'bollinger']):
                        self._validate_indicator_calculation(node, file_path, content)
                    
                    # Check backtest functions
                    if 'backtest' in node.name.lower():
                        self._validate_backtest_logic(node, file_path, content)
                    
                    # Check for numerical stability
                    self._validate_numerical_stability(node, file_path, content)
        
        except Exception as e:
            self.results.append(ValidationResult(
                passed=False,
                category='algorithms',
                rule='file_parse_error',
                details=f"Failed to parse {file_path}: {e}",
                severity='HIGH',
                file=str(file_path)
            ))
    
    def _validate_indicator_calculation(self, node: ast.FunctionDef, file_path: Path, content: str):
        """Validate indicator calculations"""
        indicator_name = node.name.lower()
        
        # Check for division by zero protection
        has_zero_check = False
        for child in ast.walk(node):
            if isinstance(child, ast.BinOp) and isinstance(child.op, ast.Div):
                # Look for zero checks before division
                if self._has_zero_protection(child, node):
                    has_zero_check = True
        
        if not has_zero_check:
            for child in ast.walk(node):
                if isinstance(child, ast.BinOp) and isinstance(child.op, ast.Div):
                    self.results.append(ValidationResult(
                        passed=False,
                        category='algorithms',
                        rule='division_by_zero',
                        details=f"Indicator '{node.name}' has unprotected division",
                        severity='HIGH',
                        file=str(file_path),
                        line=getattr(child, 'lineno', node.lineno),
                        suggestion="Add zero check: if denominator != 0:"
                    ))
                    break
    
    def _validate_backtest_logic(self, node: ast.FunctionDef, file_path: Path, content: str):
        """Validate backtesting logic"""
        backtest_requirements = {
            'has_data_validation': False,
            'has_initial_capital': False,
            'has_commission': False,
        }
        
        try:
            source = ast.unparse(node) if hasattr(ast, 'unparse') else content
        except:
            source = content
        
        # Check requirements
        if 'data' in source.lower() and ('len' in source or 'shape' in source):
            backtest_requirements['has_data_validation'] = True
        
        if 'capital' in source.lower() or 'balance' in source.lower():
            backtest_requirements['has_initial_capital'] = True
        
        if 'commission' in source.lower() or 'fee' in source.lower():
            backtest_requirements['has_commission'] = True
        
        # Critical requirements
        if not backtest_requirements['has_data_validation']:
            self.results.append(ValidationResult(
                passed=False,
                category='algorithms',
                rule='backtest_data_validation',
                details=f"Backtest '{node.name}' doesn't validate input data",
                severity='HIGH',
                file=str(file_path),
                line=node.lineno,
                suggestion="Validate data exists and has sufficient history"
            ))
    
    def _validate_numerical_stability(self, node: ast.FunctionDef, file_path: Path, content: str):
        """Check for numerical stability issues"""
        issues_found = []
        
        for child in ast.walk(node):
            # Check for float comparison
            if isinstance(child, ast.Compare):
                for op in child.ops:
                    if isinstance(op, ast.Eq):
                        # Check if comparing floats
                        left = child.left
                        if isinstance(left, ast.Name) and any(
                            t in str(left.id).lower() for t in ['price', 'float', 'value', 'rate']
                        ):
                            issues_found.append("Float equality comparison")
            
            # Check for very small/large numbers
            if isinstance(child, ast.Constant):
                if isinstance(child.value, float):
                    if 0 < abs(child.value) < 1e-10:
                        issues_found.append(f"Very small number: {child.value}")
                    elif abs(child.value) > 1e10:
                        issues_found.append(f"Very large number: {child.value}")
        
        if issues_found:
            self.results.append(ValidationResult(
                passed=False,
                category='algorithms',
                rule='numerical_stability',
                details=f"Function '{node.name}' has numerical stability issues: {', '.join(issues_found[:3])}",
                severity='MEDIUM',
                file=str(file_path),
                line=node.lineno,
                suggestion="Use np.isclose() for float comparison, check for underflow/overflow"
            ))
    
    # ========================================
    # SEMANTIC ANALYSIS
    # ========================================
    
    def _validate_semantic_analysis(self, verbose: bool):
        """Perform deep semantic analysis"""
        if verbose:
            print("\n🧬 PERFORMING SEMANTIC ANALYSIS...")
        
        # Build semantic graph
        semantic_graph = self._build_semantic_graph()
        
        # Validate data flow
        self._validate_data_flow(semantic_graph)
        
        # Validate variable usage
        self._validate_variable_usage(semantic_graph)
        
        # Validate logic flow
        self._validate_logic_flow(semantic_graph)
    
    def _build_semantic_graph(self) -> Dict:
        """Build semantic graph of entire codebase"""
        graph = {
            'functions': {},
            'variables': {},
            'dependencies': defaultdict(list),
            'data_flow': defaultdict(list),
            'call_graph': defaultdict(list)
        }
        
        # Analyze all Python files
        for py_file in Path('.').rglob('*.py'):
            if '.git' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                tree = ast.parse(content)
                
                # Build function map
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_key = f"{py_file}::{node.name}"
                        graph['functions'][func_key] = {
                            'file': str(py_file),
                            'name': node.name,
                            'line': node.lineno,
                            'args': [arg.arg for arg in node.args.args],
                            'calls': self._get_function_calls(node),
                            'variables': self._get_variables(node)
                        }
                        
                        # Build call graph
                        for called in graph['functions'][func_key]['calls']:
                            graph['call_graph'][func_key].append(called)
            
            except:
                continue
        
        return graph
    
    def _validate_data_flow(self, graph: Dict):
        """Validate data flow through the system"""
        # Check for disconnected functions
        all_functions = set(graph['functions'].keys())
        called_functions = set()
        
        for calls in graph['call_graph'].values():
            called_functions.update(calls)
        
        # Find functions that are never called (except main and test functions)
        uncalled = all_functions - called_functions
        for func in uncalled:
            if not any(skip in func for skip in ['main', 'test', '__init__']):
                self.results.append(ValidationResult(
                    passed=False,
                    category='semantic',
                    rule='uncalled_function',
                    details=f"Function {func.split('::')[1]} is never called",
                    severity='LOW',
                    file=func.split('::')[0],
                    suggestion="Remove unused function or add calls to it"
                ))
    
    def _validate_variable_usage(self, graph: Dict):
        """Validate variable usage patterns"""
        # Check for critical trading variables
        critical_vars = ['capital', 'position_size', 'stop_loss', 'risk']
        for var in critical_vars:
            found_in_code = False
            for func_info in graph['functions'].values():
                if var in func_info['variables']['assigned'] or var in func_info['variables']['used']:
                    found_in_code = True
                    break
            
            if not found_in_code:
                self.results.append(ValidationResult(
                    passed=False,
                    category='semantic',
                    rule='missing_critical_variable',
                    details=f"Critical variable '{var}' not found in codebase",
                    severity='HIGH',
                    suggestion=f"Ensure '{var}' is properly defined and used"
                ))
    
    def _validate_logic_flow(self, graph: Dict):
        """Validate logical flow consistency"""
        # Check for functions that should be connected but aren't
        trading_functions = [f for f in graph['functions'] if 'trade' in f.lower() or 'signal' in f.lower()]
        risk_functions = [f for f in graph['functions'] if 'risk' in f.lower() or 'position' in f.lower()]
        
        # Trading functions should call risk functions
        for trade_func in trading_functions:
            calls = graph['call_graph'].get(trade_func, [])
            calls_risk = any(risk_func in calls for risk_func in risk_functions)
            
            if not calls_risk and 'test' not in trade_func.lower():
                func_name = trade_func.split('::')[1]
                self.results.append(ValidationResult(
                    passed=False,
                    category='semantic',
                    rule='missing_risk_check',
                    details=f"Trading function '{func_name}' doesn't call risk management",
                    severity='CRITICAL',
                    file=trade_func.split('::')[0],
                    suggestion="Trading functions must call risk management functions"
                ))
                self.critical_issues.append(f"No risk check in {func_name}")
    
    # ========================================
    # HELPER METHODS
    # ========================================
    
    def _find_files_with_pattern(self, patterns: List[str]) -> List[Path]:
        """Find files containing specific patterns"""
        matching_files = []
        
        for py_file in Path('.').rglob('*.py'):
            if '.git' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            file_str = str(py_file).lower()
            if any(pattern in file_str for pattern in patterns):
                matching_files.append(py_file)
        
        return matching_files
    
    def _has_zero_protection(self, div_node, func_node):
        """Check if division has zero protection"""
        # Simplified check - look for if statements with != 0 or > 0
        for node in ast.walk(func_node):
            if isinstance(node, ast.If):
                for child in ast.walk(node.test):
                    if isinstance(child, ast.Compare):
                        for op in child.ops:
                            if isinstance(op, (ast.NotEq, ast.Gt)):
                                return True
        return False
    
    def _get_function_calls(self, node: ast.FunctionDef) -> List[str]:
        """Get all function calls within a function"""
        calls = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    attr = child.func.attr
                    if hasattr(child.func, 'value') and hasattr(child.func.value, 'id'):
                        calls.append(f"{child.func.value.id}.{attr}")
                    else:
                        calls.append(attr)
        
        return calls
    
    def _get_variables(self, node: ast.FunctionDef) -> Dict:
        """Get variables used in function"""
        variables = {
            'assigned': [],
            'used': []
        }
        
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        variables['assigned'].append(target.id)
            elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                variables['used'].append(child.id)
        
        return variables
    
    # ========================================
    # REPORTING
    # ========================================
    
    def _analyze_results(self) -> Dict:
        """Analyze validation results"""
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_checks': len(self.results),
            'passed': sum(1 for r in self.results if r.passed),
            'failed': sum(1 for r in self.results if not r.passed),
            'critical_issues': len(self.critical_issues),
            'by_category': {},
            'by_severity': {},
            'top_issues': [],
            'recommendations': []
        }
        
        # Count by category
        for result in self.results:
            cat = result.category
            summary['by_category'][cat] = summary['by_category'].get(cat, {'passed': 0, 'failed': 0})
            if result.passed:
                summary['by_category'][cat]['passed'] += 1
            else:
                summary['by_category'][cat]['failed'] += 1
        
        # Count by severity
        for result in self.results:
            if not result.passed:
                sev = result.severity
                summary['by_severity'][sev] = summary['by_severity'].get(sev, 0) + 1
        
        # Get top issues
        critical_results = [r for r in self.results if r.severity == 'CRITICAL' and not r.passed]
        summary['top_issues'] = [
            {
                'rule': r.rule,
                'details': r.details,
                'file': r.file,
                'suggestion': r.suggestion
            }
            for r in critical_results[:5]
        ]
        
        # Generate recommendations
        if summary['by_severity'].get('CRITICAL', 0) > 0:
            summary['recommendations'].append("URGENT: Fix all CRITICAL issues immediately")
        
        if summary['by_category'].get('risk_management', {}).get('failed', 0) > 3:
            summary['recommendations'].append("Strengthen risk management across all strategies")
        
        return summary
    
    def _save_validation_results(self, summary: Dict):
        """Save validation results to database"""
        # Save summary
        summary_path = self.validation_db.parent / 'validation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        detailed_results = [
            {
                'passed': r.passed,
                'category': r.category,
                'rule': r.rule,
                'details': r.details,
                'severity': r.severity,
                'file': r.file,
                'line': r.line,
                'suggestion': r.suggestion
            }
            for r in self.results
        ]
        
        results_path = self.validation_db.parent / f'validation_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
    
    def _print_validation_report(self, summary: Dict):
        """Print validation report"""
        print("\n" + "="*60)
        print("📊 BUSINESS LOGIC VALIDATION REPORT")
        print("="*60)
        
        # Overall stats
        print(f"\nTotal Checks: {summary['total_checks']}")
        print(f"Passed: {summary['passed']} ✅")
        print(f"Failed: {summary['failed']} ❌")
        print(f"Critical Issues: {summary['critical_issues']} 🚨")
        
        # By category
        print("\nBy Category:")
        for cat, stats in summary['by_category'].items():
            total = stats['passed'] + stats['failed']
            pass_rate = (stats['passed'] / total * 100) if total > 0 else 0
            status = "✅" if pass_rate > 80 else "⚠️" if pass_rate > 60 else "❌"
            print(f"  {status} {cat.title()}: {stats['passed']}/{total} ({pass_rate:.1f}%)")
        
        # Critical issues
        if self.critical_issues:
            print("\n🚨 CRITICAL ISSUES FOUND:")
            for issue in self.critical_issues[:10]:
                print(f"  • {issue}")
        
        # Top issues with suggestions
        if summary['top_issues']:
            print("\n🔧 TOP ISSUES TO FIX:")
            for i, issue in enumerate(summary['top_issues'], 1):
                print(f"\n{i}. {issue['rule'].upper()}")
                print(f"   File: {issue['file']}")
                print(f"   Issue: {issue['details']}")
                if issue['suggestion']:
                    print(f"   Fix: {issue['suggestion']}")
        
        # Overall health score
        health_score = (summary['passed'] / summary['total_checks'] * 100) if summary['total_checks'] > 0 else 0
        
        print(f"\n💪 BUSINESS LOGIC HEALTH SCORE: {health_score:.1f}%")
        
        if health_score >= 90:
            print("   ✅ Excellent - Business logic is solid!")
        elif health_score >= 75:
            print("   🟢 Good - Minor improvements needed")
        elif health_score >= 60:
            print("   🟡 Fair - Several issues need attention")
        else:
            print("   🔴 Poor - Critical issues must be fixed!")
        
        print("="*60)


if __name__ == "__main__":
    """Main entry point for standalone usage"""
    validator = BusinessLogicValidator()
    validator.validate_entire_bot()
