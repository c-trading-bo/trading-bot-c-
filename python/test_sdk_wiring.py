#!/usr/bin/env python3
"""
SDK Wiring Integration Tests

Validates that ML, RL, and cloud modules correctly use the SDK adapter 
for live market data, account state, and historical data access.
"""

import asyncio
import json
import sys
import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any

# Add the python directory to the path
sys.path.insert(0, os.path.dirname(__file__))
from sdk_bridge import SDKBridge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SDKWiringTests:
    """Test suite for SDK adapter wiring validation."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now(timezone.utc)
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all SDK wiring tests."""
        logger.info("🧪 Starting SDK Wiring Integration Tests...")
        
        tests = [
            self.test_sdk_bridge_basic_functionality,
            self.test_ucb_integration_with_sdk,
            self.test_decision_service_with_sdk,
            self.test_historical_data_via_sdk,
            self.test_live_market_data_access,
            self.test_account_state_retrieval,
        ]
        
        for test in tests:
            test_name = test.__name__
            try:
                logger.info(f"🔍 Running {test_name}...")
                result = await test()
                self.test_results[test_name] = {
                    'status': 'PASSED',
                    'result': result,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                logger.info(f"✅ {test_name} - PASSED")
            except Exception as e:
                self.test_results[test_name] = {
                    'status': 'FAILED',
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                logger.error(f"❌ {test_name} - FAILED: {e}")
        
        return self._generate_test_report()
    
    async def test_sdk_bridge_basic_functionality(self) -> Dict[str, Any]:
        """Test basic SDK bridge functionality."""
        async with SDKBridge(['MNQ', 'ES']) as bridge:
            # Test initialization
            assert bridge._initialized, "SDK bridge should be initialized"
            
            # Test live price retrieval
            mnq_price = await bridge.get_live_price('MNQ')
            assert mnq_price > 0, f"MNQ price should be positive, got {mnq_price}"
            
            es_price = await bridge.get_live_price('ES')
            assert es_price > 0, f"ES price should be positive, got {es_price}"
            
            # Test historical data
            bars = await bridge.get_historical_bars('MNQ', '1m', 10)
            assert len(bars) > 0, "Should return historical bars"
            assert all('timestamp' in bar for bar in bars), "All bars should have timestamps"
            
            # Test health score
            health = await bridge.get_health_score()
            assert 'health_score' in health, "Health response should include score"
            assert health['health_score'] >= 0, "Health score should be non-negative"
            
            return {
                'mnq_price': mnq_price,
                'es_price': es_price,
                'historical_bars_count': len(bars),
                'health_score': health['health_score']
            }
    
    async def test_ucb_integration_with_sdk(self) -> Dict[str, Any]:
        """Test UCB integration using SDK bridge for market data."""
        # Import UCB integration with SDK support
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ucb'))
        from neural_ucb_topstep import UCBIntegration
        
        async with UCBIntegration(['MNQ', 'ES']) as ucb:
            # Test live market features
            market_features = await ucb.get_live_market_features()
            
            required_features = ['es_price', 'nq_price', 'es_volume', 'nq_volume']
            for feature in required_features:
                assert feature in market_features, f"Market features should include {feature}"
            
            # Test strategy recommendation with live data
            strategies = ['S2', 'S3', 'S6']
            recommendation = await ucb.get_strategy_recommendation_live(strategies)
            
            assert 'trade' in recommendation, "Recommendation should include trade decision"
            assert 'market_features' in recommendation, "Recommendation should include market features"
            assert 'timestamp' in recommendation, "Recommendation should include timestamp"
            
            return {
                'market_features_keys': list(market_features.keys()),
                'recommendation_keys': list(recommendation.keys()),
                'trade_decision': recommendation.get('trade', False),
                'recommended_strategy': recommendation.get('strategy')
            }
    
    async def test_decision_service_with_sdk(self) -> Dict[str, Any]:
        """Test decision service using SDK-based integrations."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'decision_service'))
        from decision_service import DecisionService, DecisionServiceConfig
        
        config = DecisionServiceConfig(
            symbols=['MNQ', 'ES'],
            strategies=['S2', 'S3', 'S6']
        )
        
        service = DecisionService(config)
        initialized = await service.initialize()
        assert initialized, "Decision service should initialize successfully"
        
        # Test signal processing (simulated)
        signal_data = {
            'symbol': 'MNQ',
            'strategyId': 'S6',
            'side': 'long',
            'signalId': 'test_signal_001',
            'hints': {'confidence': 0.75},
            'cloud': {'p': 0.68}
        }
        
        decision = await service.on_signal(signal_data)
        
        assert 'gate' in decision, "Decision should include gate status"
        assert 'regime' in decision, "Decision should include regime information"
        
        return {
            'initialized': initialized,
            'decision_keys': list(decision.keys()),
            'gate_status': decision.get('gate', False),
            'regime': decision.get('regime')
        }
    
    async def test_historical_data_via_sdk(self) -> Dict[str, Any]:
        """Test historical data access via SDK adapter."""
        async with SDKBridge(['MNQ']) as bridge:
            # Test different timeframes
            timeframes = ['1m', '5m']
            results = {}
            
            for tf in timeframes:
                bars = await bridge.get_historical_bars('MNQ', tf, 20)
                results[tf] = {
                    'count': len(bars),
                    'has_ohlcv': all(
                        'open' in bar and 'high' in bar and 'low' in bar and 
                        'close' in bar and 'volume' in bar 
                        for bar in bars
                    ) if bars else False
                }
            
            return results
    
    async def test_live_market_data_access(self) -> Dict[str, Any]:
        """Test live market data access across multiple instruments."""
        async with SDKBridge(['MNQ', 'ES', 'RTY']) as bridge:
            prices = {}
            
            instruments = ['MNQ', 'ES', 'RTY']
            for instrument in instruments:
                try:
                    price = await bridge.get_live_price(instrument)
                    prices[instrument] = {
                        'price': price,
                        'valid': price > 0
                    }
                except Exception as e:
                    prices[instrument] = {
                        'error': str(e),
                        'valid': False
                    }
            
            return prices
    
    async def test_account_state_retrieval(self) -> Dict[str, Any]:
        """Test account state and portfolio information retrieval."""
        async with SDKBridge() as bridge:
            account_state = await bridge.get_account_state()
            
            required_keys = ['portfolio', 'health', 'timestamp']
            for key in required_keys:
                assert key in account_state, f"Account state should include {key}"
            
            # Test portfolio structure
            portfolio = account_state['portfolio']
            assert 'positions' in account_state, "Account state should include positions"
            
            # Test health information
            health = account_state['health']
            assert 'health_score' in health, "Health should include score"
            
            return {
                'account_keys': list(account_state.keys()),
                'portfolio_keys': list(portfolio.keys()) if isinstance(portfolio, dict) else [],
                'health_score': health.get('health_score', 0),
                'positions_count': len(account_state.get('positions', {}))
            }
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        end_time = datetime.now(timezone.utc)
        duration = (end_time - self.start_time).total_seconds()
        
        passed_tests = [name for name, result in self.test_results.items() if result['status'] == 'PASSED']
        failed_tests = [name for name, result in self.test_results.items() if result['status'] == 'FAILED']
        
        report = {
            'test_summary': {
                'total_tests': len(self.test_results),
                'passed': len(passed_tests),
                'failed': len(failed_tests),
                'success_rate': len(passed_tests) / len(self.test_results) * 100 if self.test_results else 0,
                'duration_seconds': duration
            },
            'test_results': self.test_results,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'execution_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'environment': 'SDK Wiring Integration Test'
            }
        }
        
        return report

async def main():
    """Run SDK wiring integration tests."""
    print("🚀 SDK Wiring Integration Test Suite")
    print("=" * 50)
    
    tester = SDKWiringTests()
    report = await tester.run_all_tests()
    
    # Print summary
    summary = report['test_summary']
    print(f"\n📊 Test Results Summary:")
    print(f"   Total Tests: {summary['total_tests']}")
    print(f"   Passed: {summary['passed']}")
    print(f"   Failed: {summary['failed']}")
    print(f"   Success Rate: {summary['success_rate']:.1f}%")
    print(f"   Duration: {summary['duration_seconds']:.2f}s")
    
    if report['failed_tests']:
        print(f"\n❌ Failed Tests:")
        for test_name in report['failed_tests']:
            error = report['test_results'][test_name]['error']
            print(f"   - {test_name}: {error}")
    
    if report['passed_tests']:
        print(f"\n✅ Passed Tests:")
        for test_name in report['passed_tests']:
            print(f"   - {test_name}")
    
    # Save detailed report
    report_file = f"sdk_wiring_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Return appropriate exit code
    return 0 if summary['failed'] == 0 else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)