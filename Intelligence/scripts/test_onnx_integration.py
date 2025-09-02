#!/usr/bin/env python3
"""
ONNX Integration Testing Suite
Enhanced ML infrastructure testing for institutional-grade AI systems
"""

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

class OnnxTester:
    """Comprehensive ONNX model testing and validation suite"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logger()
        self.test_results = {}
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("OnnxTester")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def create_test_model(self) -> str:
        """Create a test ONNX model for validation"""
        try:
            # Simple neural network for testing
            class TestModel(nn.Module):
                def __init__(self):
                    super(TestModel, self).__init__()
                    self.layers = nn.Sequential(
                        nn.Linear(30, 64),  # 30 features input
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 4),   # 4 outputs for position sizing
                        nn.Softmax(dim=1)
                    )
                
                def forward(self, x):
                    return self.layers(x)
            
            # Create model and sample input
            model = TestModel()
            model.eval()
            
            # Sample input (batch_size=1, features=30)
            sample_input = torch.randn(1, 30)
            
            # Export to ONNX
            onnx_path = self.models_dir / "test_rl_model.onnx"
            torch.onnx.export(
                model,
                sample_input,
                str(onnx_path),
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=['features'],
                output_names=['position_logits'],
                dynamic_axes={
                    'features': {0: 'batch_size'},
                    'position_logits': {0: 'batch_size'}
                }
            )
            
            self.logger.info(f"Test ONNX model created: {onnx_path}")
            return str(onnx_path)
            
        except Exception as e:
            self.logger.error(f"Error creating test model: {e}")
            raise
    
    def test_onnx_model_loading(self, model_path: str) -> bool:
        """Test ONNX model loading and validation"""
        try:
            self.logger.info(f"Testing ONNX model loading: {model_path}")
            
            # Load and validate ONNX model
            onnx_model = onnx.load(model_path)
            onnx.checker.check_model(onnx_model)
            
            # Test with ONNX Runtime
            session = ort.InferenceSession(model_path)
            
            # Get input/output info
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            
            self.logger.info(f"✅ Model loaded successfully")
            self.logger.info(f"   Input: {input_info.name} - {input_info.shape}")
            self.logger.info(f"   Output: {output_info.name} - {output_info.shape}")
            
            self.test_results['model_loading'] = {
                'success': True,
                'input_shape': input_info.shape,
                'output_shape': output_info.shape
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model loading failed: {e}")
            self.test_results['model_loading'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def test_model_inference(self, model_path: str, num_tests: int = 100) -> bool:
        """Test model inference performance and accuracy"""
        try:
            self.logger.info(f"Testing model inference ({num_tests} iterations)")
            
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            
            inference_times = []
            
            for i in range(num_tests):
                # Generate random test input (30 features)
                test_input = np.random.randn(1, 30).astype(np.float32)
                
                # Measure inference time
                import time
                start_time = time.perf_counter()
                
                output = session.run(None, {input_name: test_input})
                
                end_time = time.perf_counter()
                inference_times.append((end_time - start_time) * 1000)  # ms
                
                # Validate output
                if len(output) != 1:
                    raise ValueError(f"Expected 1 output, got {len(output)}")
                
                prediction = output[0]
                if prediction.shape != (1, 4):
                    raise ValueError(f"Expected output shape (1, 4), got {prediction.shape}")
                
                # Check if probabilities sum to ~1 (softmax output)
                prob_sum = np.sum(prediction)
                if not (0.99 <= prob_sum <= 1.01):
                    raise ValueError(f"Softmax output sum invalid: {prob_sum}")
            
            avg_inference_time = np.mean(inference_times)
            max_inference_time = np.max(inference_times)
            
            self.logger.info(f"✅ Inference test passed")
            self.logger.info(f"   Average inference time: {avg_inference_time:.2f}ms")
            self.logger.info(f"   Maximum inference time: {max_inference_time:.2f}ms")
            
            self.test_results['inference'] = {
                'success': True,
                'avg_time_ms': avg_inference_time,
                'max_time_ms': max_inference_time,
                'tests_run': num_tests
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Inference test failed: {e}")
            self.test_results['inference'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def test_batch_inference(self, model_path: str, batch_sizes: List[int] = [1, 8, 16, 32]) -> bool:
        """Test batch inference capabilities"""
        try:
            self.logger.info(f"Testing batch inference for sizes: {batch_sizes}")
            
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            
            batch_results = {}
            
            for batch_size in batch_sizes:
                # Generate batch input
                batch_input = np.random.randn(batch_size, 30).astype(np.float32)
                
                # Run inference
                import time
                start_time = time.perf_counter()
                output = session.run(None, {input_name: batch_input})
                end_time = time.perf_counter()
                
                inference_time = (end_time - start_time) * 1000  # ms
                
                # Validate output
                prediction = output[0]
                expected_shape = (batch_size, 4)
                if prediction.shape != expected_shape:
                    raise ValueError(f"Batch {batch_size}: Expected {expected_shape}, got {prediction.shape}")
                
                batch_results[batch_size] = {
                    'time_ms': inference_time,
                    'time_per_sample_ms': inference_time / batch_size
                }
                
                self.logger.info(f"   Batch {batch_size}: {inference_time:.2f}ms "
                               f"({inference_time/batch_size:.2f}ms/sample)")
            
            self.test_results['batch_inference'] = {
                'success': True,
                'results': batch_results
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Batch inference test failed: {e}")
            self.test_results['batch_inference'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def test_model_optimization(self, model_path: str) -> bool:
        """Test different ONNX optimization levels"""
        try:
            self.logger.info("Testing ONNX optimization levels")
            
            # Test different session options
            optimization_levels = [
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            ]
            
            optimization_results = {}
            test_input = np.random.randn(1, 30).astype(np.float32)
            
            for level in optimization_levels:
                try:
                    session_options = ort.SessionOptions()
                    session_options.graph_optimization_level = level
                    
                    session = ort.InferenceSession(model_path, session_options)
                    input_name = session.get_inputs()[0].name
                    
                    # Time multiple inferences
                    import time
                    times = []
                    for _ in range(50):
                        start = time.perf_counter()
                        session.run(None, {input_name: test_input})
                        times.append((time.perf_counter() - start) * 1000)
                    
                    avg_time = np.mean(times)
                    optimization_results[level.name] = avg_time
                    
                    self.logger.info(f"   {level.name}: {avg_time:.2f}ms")
                    
                except Exception as e:
                    self.logger.warning(f"   {level.name}: Failed - {e}")
                    optimization_results[level.name] = None
            
            self.test_results['optimization'] = {
                'success': True,
                'results': optimization_results
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Optimization test failed: {e}")
            self.test_results['optimization'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def test_feature_engineering_compatibility(self, model_path: str) -> bool:
        """Test model with realistic trading features"""
        try:
            self.logger.info("Testing feature engineering compatibility")
            
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            
            # Create realistic trading features
            features = self._generate_realistic_features()
            
            # Test each feature set
            for scenario, feature_data in features.items():
                try:
                    output = session.run(None, {input_name: feature_data})
                    prediction = output[0]
                    
                    # Validate prediction makes sense
                    if not np.all(prediction >= 0) or not np.all(prediction <= 1):
                        raise ValueError(f"Invalid probabilities in {scenario}")
                    
                    if not (0.99 <= np.sum(prediction) <= 1.01):
                        raise ValueError(f"Probabilities don't sum to 1 in {scenario}")
                    
                    self.logger.info(f"   ✅ {scenario}: {prediction[0]}")
                    
                except Exception as e:
                    raise ValueError(f"Scenario {scenario} failed: {e}")
            
            self.test_results['feature_compatibility'] = {
                'success': True,
                'scenarios_tested': list(features.keys())
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Feature compatibility test failed: {e}")
            self.test_results['feature_compatibility'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def _generate_realistic_features(self) -> Dict[str, np.ndarray]:
        """Generate realistic trading feature scenarios"""
        scenarios = {}
        
        # Bull market scenario
        bull_features = np.array([
            0.8,   # momentum_1m
            0.75,  # momentum_5m
            0.7,   # momentum_15m
            0.9,   # volume_ratio
            0.6,   # volatility
            0.2,   # vix_level (normalized)
            0.8,   # price_above_vwap
            0.9,   # rsi_normalized
            0.7,   # macd_signal
            0.8,   # ema_cross_signal
            # Additional features (20 more)
            *np.random.rand(20) * 0.5 + 0.25  # Mix of moderate signals
        ]).reshape(1, 30).astype(np.float32)
        
        # Bear market scenario
        bear_features = np.array([
            0.1,   # momentum_1m (negative)
            0.15,  # momentum_5m
            0.2,   # momentum_15m
            1.2,   # volume_ratio (high selling volume)
            0.9,   # volatility (high)
            0.8,   # vix_level (high fear)
            0.1,   # price_above_vwap (below)
            0.2,   # rsi_normalized (oversold)
            0.1,   # macd_signal (bearish)
            0.2,   # ema_cross_signal (bearish)
            # Additional features
            *np.random.rand(20) * 0.3  # Low to moderate signals
        ]).reshape(1, 30).astype(np.float32)
        
        # Sideways market scenario
        sideways_features = np.array([
            0.5,   # momentum_1m (neutral)
            0.48,  # momentum_5m
            0.52,  # momentum_15m
            0.5,   # volume_ratio
            0.3,   # volatility (low)
            0.4,   # vix_level (moderate)
            0.5,   # price_above_vwap
            0.5,   # rsi_normalized (neutral)
            0.5,   # macd_signal
            0.5,   # ema_cross_signal
            # Additional features
            *np.random.rand(20) * 0.6 + 0.2  # Random moderate signals
        ]).reshape(1, 30).astype(np.float32)
        
        scenarios["bull_market"] = bull_features
        scenarios["bear_market"] = bear_features  
        scenarios["sideways_market"] = sideways_features
        
        return scenarios
    
    def run_comprehensive_test(self) -> Dict:
        """Run all ONNX integration tests"""
        self.logger.info("🧪 Starting comprehensive ONNX integration tests")
        self.logger.info("=" * 60)
        
        try:
            # Create test model
            test_model_path = self.create_test_model()
            
            # Run all tests
            tests_passed = 0
            total_tests = 5
            
            if self.test_onnx_model_loading(test_model_path):
                tests_passed += 1
                
            if self.test_model_inference(test_model_path):
                tests_passed += 1
                
            if self.test_batch_inference(test_model_path):
                tests_passed += 1
                
            if self.test_model_optimization(test_model_path):
                tests_passed += 1
                
            if self.test_feature_engineering_compatibility(test_model_path):
                tests_passed += 1
            
            # Summary
            self.logger.info("=" * 60)
            self.logger.info(f"🎯 ONNX Integration Tests Complete: {tests_passed}/{total_tests} passed")
            
            if tests_passed == total_tests:
                self.logger.info("✅ All ONNX integration tests PASSED!")
                self.test_results['overall'] = {'success': True, 'tests_passed': tests_passed}
            else:
                self.logger.warning(f"⚠️  {total_tests - tests_passed} tests FAILED")
                self.test_results['overall'] = {'success': False, 'tests_passed': tests_passed}
            
            # Save test results
            self._save_test_results()
            
            return self.test_results
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive test failed: {e}")
            self.test_results['overall'] = {'success': False, 'error': str(e)}
            return self.test_results
    
    def _save_test_results(self):
        """Save test results to JSON file"""
        try:
            results_file = self.models_dir / "onnx_test_results.json"
            with open(results_file, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            self.logger.info(f"Test results saved to {results_file}")
        except Exception as e:
            self.logger.error(f"Error saving test results: {e}")


def main():
    """Main function for standalone execution"""
    tester = OnnxTester()
    results = tester.run_comprehensive_test()
    
    # Exit with appropriate code
    if results.get('overall', {}).get('success', False):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()