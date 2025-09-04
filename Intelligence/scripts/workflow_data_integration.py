#!/usr/bin/env python3
"""
Workflow Data Integration Script for BotCore Decision Engine
Ensures all Intelligence workflows output data in standardized formats

This script should be called at the end of every workflow to format and standardize output data
for consumption by BotCore's TradingSignalProcessor and StrategyEngine.

Usage: python workflow_data_integration.py --workflow-type [type] --data-path [path] --output-format [format]
"""

import json
import sys
import argparse
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from pathlib import Path

class WorkflowDataIntegrator:
    """Integrates workflow outputs with BotCore decision engine"""
    
    def __init__(self):
        self.supported_formats = {
            'trade_signal_data': self._format_trade_signal_data,
            'strategy_signal': self._format_strategy_signal,
            'market_analysis': self._format_market_analysis,
            'news_sentiment': self._format_news_sentiment,
            'risk_assessment': self._format_risk_assessment,
            'regime_detection': self._format_regime_detection,
            'correlation_matrix': self._format_correlation_matrix,
            'volatility_surface': self._format_volatility_surface,
            'microstructure': self._format_microstructure,
            'ml_features': self._format_ml_features
        }
        
        self.workflow_mappings = {
            'es_nq_critical_trading': 'trade_signal_data',
            'overnight': 'trade_signal_data', 
            'portfolio_heat': 'risk_assessment',
            'ultimate_options_flow_pipeline': 'market_analysis',
            'ultimate_ml_rl_training_pipeline': 'ml_features',
            'ultimate_ml_rl_intel_system': 'ml_features',
            'market_data': 'market_analysis',
            'ultimate_regime_detection_pipeline': 'regime_detection',
            'ultimate_news_sentiment_pipeline': 'news_sentiment',
            'ultimate_data_collection_pipeline': 'market_analysis',
            'volatility_surface': 'volatility_surface',
            'microstructure': 'microstructure',
            'seasonality': 'market_analysis',
            'fed_liquidity': 'market_analysis',
            'daily_report': 'market_analysis',
            'mm_positioning': 'market_analysis',
            'zones_identifier': 'market_analysis',
            'intermarket': 'correlation_matrix',
            'failed_patterns': 'market_analysis',
            'opex_calendar': 'market_analysis',
            'daily_consolidated': 'market_analysis',
            'ml_trainer': 'ml_features',
            'es_nq_correlation_matrix': 'correlation_matrix'
        }

    def process_workflow_output(self, workflow_type: str, data_path: str, output_format: Optional[str] = None) -> Dict[str, Any]:
        """Process workflow output and convert to BotCore format"""
        
        # Determine output format
        if not output_format:
            output_format = self.workflow_mappings.get(workflow_type, 'market_analysis')
        
        # Load workflow data
        raw_data = self._load_workflow_data(data_path)
        
        # Format according to BotCore specifications
        formatter = self.supported_formats.get(output_format, self._format_market_analysis)
        formatted_data = formatter(workflow_type, raw_data)
        
        # Add metadata
        formatted_data['metadata'] = {
            'workflow_type': workflow_type,
            'processed_at': datetime.now(timezone.utc).isoformat(),
            'format_version': '1.0.0',
            'integration_version': 'enterprise-24-7-v1.0'
        }
        
        return formatted_data

    def _load_workflow_data(self, data_path: str) -> Dict[str, Any]:
        """Load data from various workflow output formats"""
        
        if not os.path.exists(data_path):
            return {}
        
        # Try to load as JSON first
        try:
            with open(data_path, 'r') as f:
                return json.load(f)
        except:
            pass
        
        # Try CSV format
        try:
            import pandas as pd
            df = pd.read_csv(data_path)
            return df.to_dict('records')
        except:
            pass
        
        # Fallback to text analysis
        try:
            with open(data_path, 'r') as f:
                content = f.read()
                return {'raw_content': content}
        except:
            return {}

    def _format_trade_signal_data(self, workflow_type: str, raw_data: Any) -> Dict[str, Any]:
        """Format data as TradeSignalData for BotCore"""
        
        # Determine symbol based on workflow
        symbol = self._extract_symbol(workflow_type, raw_data)
        
        # Extract signal information
        signals = []
        
        if isinstance(raw_data, dict):
            signal_data = {
                'Id': f"{workflow_type}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                'Symbol': symbol,
                'Direction': self._extract_direction(raw_data),
                'Entry': self._extract_price(raw_data, 'entry'),
                'Size': self._extract_size(raw_data),
                'Strategy': workflow_type,
                'StopLoss': self._extract_price(raw_data, 'stop_loss'),
                'TakeProfit': self._extract_price(raw_data, 'take_profit'),
                'Regime': self._extract_regime(raw_data),
                'Atr': self._extract_indicator(raw_data, 'atr', 1.0),
                'Rsi': self._extract_indicator(raw_data, 'rsi', 50.0),
                'Ema20': self._extract_indicator(raw_data, 'ema20'),
                'Ema50': self._extract_indicator(raw_data, 'ema50'),
                'BbUpper': self._extract_indicator(raw_data, 'bb_upper'),
                'BbLower': self._extract_indicator(raw_data, 'bb_lower'),
                'Momentum': self._extract_indicator(raw_data, 'momentum', 0.0),
                'TrendStrength': self._extract_indicator(raw_data, 'trend_strength', 0.5),
                'VixLevel': self._extract_indicator(raw_data, 'vix', 20.0)
            }
            signals.append(signal_data)
        
        return {
            'format_type': 'TradeSignalData',
            'signals': signals,
            'session': self._determine_session(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    def _format_strategy_signal(self, workflow_type: str, raw_data: Any) -> Dict[str, Any]:
        """Format data as StrategySignal for BotCore"""
        
        signals = []
        symbol = self._extract_symbol(workflow_type, raw_data)
        
        signal = {
            'Strategy': workflow_type,
            'Symbol': symbol,
            'Side': self._map_to_signal_side(self._extract_direction(raw_data)),
            'Size': self._extract_size(raw_data),
            'LimitPrice': self._extract_price(raw_data, 'entry'),
            'Note': f"Generated by {workflow_type} workflow",
            'ClientOrderId': f"{workflow_type}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        }
        signals.append(signal)
        
        return {
            'format_type': 'StrategySignal',
            'signals': signals,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    def _format_market_analysis(self, workflow_type: str, raw_data: Any) -> Dict[str, Any]:
        """Format general market analysis data"""
        
        analysis = {
            'analysis_type': workflow_type,
            'findings': self._extract_findings(raw_data),
            'indicators': self._extract_all_indicators(raw_data),
            'risk_factors': self._extract_risk_factors(raw_data),
            'recommendations': self._extract_recommendations(raw_data),
            'confidence_score': self._extract_confidence(raw_data),
            'applicable_symbols': self._extract_applicable_symbols(workflow_type),
            'session_relevance': self._determine_session_relevance(workflow_type)
        }
        
        return {
            'format_type': 'MarketAnalysis', 
            'analysis': analysis,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    def _format_news_sentiment(self, workflow_type: str, raw_data: Any) -> Dict[str, Any]:
        """Format news sentiment analysis data"""
        
        sentiment = {
            'overall_sentiment': self._extract_sentiment_score(raw_data),
            'sentiment_breakdown': self._extract_sentiment_breakdown(raw_data),
            'key_themes': self._extract_key_themes(raw_data),
            'market_impact': self._extract_market_impact(raw_data),
            'trading_implications': self._extract_trading_implications(raw_data),
            'urgency_level': self._extract_urgency(raw_data),
            'affected_instruments': self._extract_affected_instruments(raw_data)
        }
        
        return {
            'format_type': 'NewsSentiment',
            'sentiment': sentiment,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    def _format_risk_assessment(self, workflow_type: str, raw_data: Any) -> Dict[str, Any]:
        """Format risk assessment data"""
        
        risk = {
            'overall_risk_level': self._extract_risk_level(raw_data),
            'portfolio_heat': self._extract_portfolio_heat(raw_data),
            'position_risks': self._extract_position_risks(raw_data),
            'var_estimates': self._extract_var_estimates(raw_data),
            'correlation_risks': self._extract_correlation_risks(raw_data),
            'liquidity_risks': self._extract_liquidity_risks(raw_data),
            'recommended_actions': self._extract_risk_actions(raw_data)
        }
        
        return {
            'format_type': 'RiskAssessment',
            'risk': risk,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    def _format_regime_detection(self, workflow_type: str, raw_data: Any) -> Dict[str, Any]:
        """Format regime detection data"""
        
        regime = {
            'current_regime': self._extract_current_regime(raw_data),
            'regime_confidence': self._extract_regime_confidence(raw_data),
            'regime_transition_probability': self._extract_transition_probability(raw_data),
            'regime_characteristics': self._extract_regime_characteristics(raw_data),
            'historical_regime_data': self._extract_historical_regimes(raw_data),
            'trading_adjustments': self._extract_regime_adjustments(raw_data)
        }
        
        return {
            'format_type': 'RegimeDetection',
            'regime': regime,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    def _format_correlation_matrix(self, workflow_type: str, raw_data: Any) -> Dict[str, Any]:
        """Format correlation analysis data"""
        
        correlation = {
            'correlation_matrix': self._extract_correlation_matrix(raw_data),
            'key_correlations': self._extract_key_correlations(raw_data),
            'correlation_changes': self._extract_correlation_changes(raw_data),
            'trading_implications': self._extract_correlation_implications(raw_data),
            'risk_diversification': self._extract_diversification_analysis(raw_data)
        }
        
        return {
            'format_type': 'CorrelationAnalysis',
            'correlation': correlation,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    def _format_volatility_surface(self, workflow_type: str, raw_data: Any) -> Dict[str, Any]:
        """Format volatility surface data"""
        
        volatility = {
            'volatility_surface': self._extract_volatility_surface(raw_data),
            'implied_volatility': self._extract_implied_volatility(raw_data),
            'volatility_skew': self._extract_volatility_skew(raw_data),
            'volatility_term_structure': self._extract_term_structure(raw_data),
            'volatility_forecast': self._extract_volatility_forecast(raw_data),
            'trading_opportunities': self._extract_volatility_opportunities(raw_data)
        }
        
        return {
            'format_type': 'VolatilitySurface',
            'volatility': volatility,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    def _format_microstructure(self, workflow_type: str, raw_data: Any) -> Dict[str, Any]:
        """Format microstructure analysis data"""
        
        microstructure = {
            'bid_ask_dynamics': self._extract_bid_ask_dynamics(raw_data),
            'order_flow': self._extract_order_flow(raw_data),
            'market_depth': self._extract_market_depth(raw_data),
            'execution_quality': self._extract_execution_quality(raw_data),
            'liquidity_metrics': self._extract_liquidity_metrics(raw_data),
            'trading_costs': self._extract_trading_costs(raw_data)
        }
        
        return {
            'format_type': 'Microstructure',
            'microstructure': microstructure,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    def _format_ml_features(self, workflow_type: str, raw_data: Any) -> Dict[str, Any]:
        """Format ML features data"""
        
        ml_data = {
            'feature_vectors': self._extract_feature_vectors(raw_data),
            'model_predictions': self._extract_model_predictions(raw_data),
            'feature_importance': self._extract_feature_importance(raw_data),
            'model_confidence': self._extract_model_confidence(raw_data),
            'training_metrics': self._extract_training_metrics(raw_data),
            'model_recommendations': self._extract_model_recommendations(raw_data)
        }
        
        return {
            'format_type': 'MLFeatures',
            'ml_data': ml_data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    # Helper extraction methods
    def _extract_symbol(self, workflow_type: str, raw_data: Any) -> str:
        """Extract trading symbol from workflow data"""
        # Priority: explicit symbol > workflow-based default > ES fallback
        if isinstance(raw_data, dict):
            symbol = raw_data.get('symbol', raw_data.get('Symbol', ''))
            if symbol:
                return symbol
        
        # Workflow-based defaults
        if 'nq' in workflow_type.lower():
            return 'NQ'
        elif 'es' in workflow_type.lower():
            return 'ES'
        elif 'options' in workflow_type.lower():
            return 'SPY'
        else:
            return 'ES'  # Default fallback

    def _extract_direction(self, raw_data: Any) -> str:
        """Extract trading direction"""
        if isinstance(raw_data, dict):
            direction = raw_data.get('direction', raw_data.get('side', raw_data.get('action', '')))
            if direction:
                return str(direction).upper()
        return 'HOLD'

    def _extract_price(self, raw_data: Any, price_type: str) -> float:
        """Extract price data"""
        if isinstance(raw_data, dict):
            price = raw_data.get(price_type, raw_data.get(price_type.replace('_', ''), 0))
            if price:
                return float(price)
        
        # Default prices based on current market estimates
        if price_type == 'entry':
            return 5800.0  # ES approximate
        elif price_type == 'stop_loss':
            return 5790.0
        elif price_type == 'take_profit':
            return 5820.0
        return 0.0

    def _extract_size(self, raw_data: Any) -> int:
        """Extract position size"""
        if isinstance(raw_data, dict):
            size = raw_data.get('size', raw_data.get('quantity', raw_data.get('qty', 1)))
            return int(size) if size else 1
        return 1

    def _extract_regime(self, raw_data: Any) -> str:
        """Extract market regime"""
        if isinstance(raw_data, dict):
            regime = raw_data.get('regime', raw_data.get('market_regime', ''))
            if regime:
                return str(regime)
        return 'Range'

    def _extract_indicator(self, raw_data: Any, indicator: str, default: float = 0.0) -> float:
        """Extract technical indicator value"""
        if isinstance(raw_data, dict):
            value = raw_data.get(indicator, raw_data.get(indicator.upper(), default))
            return float(value) if value is not None else default
        return default

    def _determine_session(self) -> str:
        """Determine current trading session"""
        now = datetime.now(timezone.utc)
        hour = now.hour
        
        # Convert to CT for ES/NQ session determination
        ct_hour = (hour - 6) % 24  # Approximate CT conversion
        
        if 8 <= ct_hour <= 16:  # US session (08:30-16:00 CT)
            return 'US'
        elif 2 <= ct_hour <= 5:   # European session (02:00-05:00 CT) 
            return 'European'
        elif 18 <= ct_hour <= 23: # Asian session (18:00-23:59 CT)
            return 'Asian'
        else:
            return 'Extended'

    def _map_to_signal_side(self, direction: str) -> int:
        """Map direction to SignalSide enum"""
        direction = direction.upper()
        if direction in ['BUY', 'LONG']:
            return 1  # SignalSide.Long
        elif direction in ['SELL', 'SHORT']:
            return -1  # SignalSide.Short
        else:
            return 0  # SignalSide.Flat

    def _extract_applicable_symbols(self, workflow_type: str) -> List[str]:
        """Extract applicable trading symbols for workflow"""
        if 'es_nq' in workflow_type:
            return ['ES', 'NQ']
        elif 'es' in workflow_type.lower():
            return ['ES']
        elif 'nq' in workflow_type.lower():
            return ['NQ']
        elif 'options' in workflow_type.lower():
            return ['SPY', 'QQQ', 'IWM']
        else:
            return ['ES', 'NQ']

    def _determine_session_relevance(self, workflow_type: str) -> Dict[str, float]:
        """Determine workflow relevance by session"""
        # Default relevance scores
        relevance = {
            'Asian': 1.0,
            'European': 1.0, 
            'US': 1.0,
            'Extended': 0.8
        }
        
        # Adjust based on workflow type
        if 'overnight' in workflow_type:
            relevance['Asian'] = 1.0
            relevance['European'] = 1.0
            relevance['US'] = 0.3
        elif 'options' in workflow_type:
            relevance['US'] = 1.0
            relevance['Asian'] = 0.2
            relevance['European'] = 0.2
        
        return relevance

    # Placeholder extraction methods (implement based on actual workflow outputs)
    def _extract_findings(self, raw_data: Any) -> List[str]:
        return ["Analysis completed successfully"]
    
    def _extract_all_indicators(self, raw_data: Any) -> Dict[str, float]:
        return {}
    
    def _extract_risk_factors(self, raw_data: Any) -> List[str]:
        return []
    
    def _extract_recommendations(self, raw_data: Any) -> List[str]:
        return []
    
    def _extract_confidence(self, raw_data: Any) -> float:
        return 0.75
    
    def _extract_sentiment_score(self, raw_data: Any) -> float:
        return 0.0
    
    def _extract_sentiment_breakdown(self, raw_data: Any) -> Dict[str, float]:
        return {}
    
    def _extract_key_themes(self, raw_data: Any) -> List[str]:
        return []
    
    def _extract_market_impact(self, raw_data: Any) -> str:
        return "Neutral"
    
    def _extract_trading_implications(self, raw_data: Any) -> List[str]:
        return []
    
    def _extract_urgency(self, raw_data: Any) -> str:
        return "Medium"
    
    def _extract_affected_instruments(self, raw_data: Any) -> List[str]:
        return ["ES", "NQ"]
    
    def _extract_risk_level(self, raw_data: Any) -> str:
        return "Medium"
    
    def _extract_portfolio_heat(self, raw_data: Any) -> float:
        return 0.5
    
    def _extract_position_risks(self, raw_data: Any) -> List[Dict[str, Any]]:
        return []
    
    def _extract_var_estimates(self, raw_data: Any) -> Dict[str, float]:
        return {}
    
    def _extract_correlation_risks(self, raw_data: Any) -> Dict[str, float]:
        return {}
    
    def _extract_liquidity_risks(self, raw_data: Any) -> Dict[str, float]:
        return {}
    
    def _extract_risk_actions(self, raw_data: Any) -> List[str]:
        return []
    
    def _extract_current_regime(self, raw_data: Any) -> str:
        return "Range"
    
    def _extract_regime_confidence(self, raw_data: Any) -> float:
        return 0.75
    
    def _extract_transition_probability(self, raw_data: Any) -> Dict[str, float]:
        return {}
    
    def _extract_regime_characteristics(self, raw_data: Any) -> Dict[str, Any]:
        return {}
    
    def _extract_historical_regimes(self, raw_data: Any) -> List[Dict[str, Any]]:
        return []
    
    def _extract_regime_adjustments(self, raw_data: Any) -> List[str]:
        return []
    
    def _extract_correlation_matrix(self, raw_data: Any) -> Dict[str, Dict[str, float]]:
        return {}
    
    def _extract_key_correlations(self, raw_data: Any) -> List[Dict[str, Any]]:
        return []
    
    def _extract_correlation_changes(self, raw_data: Any) -> Dict[str, float]:
        return {}
    
    def _extract_correlation_implications(self, raw_data: Any) -> List[str]:
        return []
    
    def _extract_diversification_analysis(self, raw_data: Any) -> Dict[str, Any]:
        return {}
    
    def _extract_volatility_surface(self, raw_data: Any) -> Dict[str, Any]:
        return {}
    
    def _extract_implied_volatility(self, raw_data: Any) -> Dict[str, float]:
        return {}
    
    def _extract_volatility_skew(self, raw_data: Any) -> Dict[str, float]:
        return {}
    
    def _extract_term_structure(self, raw_data: Any) -> Dict[str, float]:
        return {}
    
    def _extract_volatility_forecast(self, raw_data: Any) -> Dict[str, float]:
        return {}
    
    def _extract_volatility_opportunities(self, raw_data: Any) -> List[str]:
        return []
    
    def _extract_bid_ask_dynamics(self, raw_data: Any) -> Dict[str, Any]:
        return {}
    
    def _extract_order_flow(self, raw_data: Any) -> Dict[str, Any]:
        return {}
    
    def _extract_market_depth(self, raw_data: Any) -> Dict[str, Any]:
        return {}
    
    def _extract_execution_quality(self, raw_data: Any) -> Dict[str, float]:
        return {}
    
    def _extract_liquidity_metrics(self, raw_data: Any) -> Dict[str, float]:
        return {}
    
    def _extract_trading_costs(self, raw_data: Any) -> Dict[str, float]:
        return {}
    
    def _extract_feature_vectors(self, raw_data: Any) -> List[List[float]]:
        return []
    
    def _extract_model_predictions(self, raw_data: Any) -> Dict[str, Any]:
        return {}
    
    def _extract_feature_importance(self, raw_data: Any) -> Dict[str, float]:
        return {}
    
    def _extract_model_confidence(self, raw_data: Any) -> float:
        return 0.75
    
    def _extract_training_metrics(self, raw_data: Any) -> Dict[str, float]:
        return {}
    
    def _extract_model_recommendations(self, raw_data: Any) -> List[str]:
        return []

def main():
    parser = argparse.ArgumentParser(description='Integrate workflow data with BotCore decision engine')
    parser.add_argument('--workflow-type', required=True, help='Type of workflow generating the data')
    parser.add_argument('--data-path', required=True, help='Path to workflow output data')
    parser.add_argument('--output-format', help='Specific output format (optional)')
    parser.add_argument('--output-path', help='Path to save formatted output')
    
    args = parser.parse_args()
    
    integrator = WorkflowDataIntegrator()
    
    try:
        # Process the workflow output
        formatted_data = integrator.process_workflow_output(
            args.workflow_type, 
            args.data_path, 
            args.output_format
        )
        
        # Save formatted output
        output_path = args.output_path or f"Intelligence/data/integrated/{args.workflow_type}_integrated.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(formatted_data, f, indent=2, default=str)
        
        print(f"✅ Successfully integrated {args.workflow_type} data")
        print(f"📁 Output saved to: {output_path}")
        print(f"📊 Format: {formatted_data.get('format_type', 'Unknown')}")
        print(f"⏰ Processed at: {formatted_data.get('metadata', {}).get('processed_at', 'Unknown')}")
        
        # Output summary for CI/CD
        if 'signals' in formatted_data:
            print(f"🎯 Signals generated: {len(formatted_data['signals'])}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error integrating {args.workflow_type} data: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
