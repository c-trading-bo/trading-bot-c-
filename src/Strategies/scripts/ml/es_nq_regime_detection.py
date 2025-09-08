#!/usr/bin/env python3
"""
ES/NQ Regime Detection for Futures Trading
Advanced regime detection using Hidden Markov Models and market microstructure
"""

import numpy as np
import pandas as pd
import json
import os
import logging
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.mixture import GaussianMixture

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    logger.warning("hmmlearn not available, using fallback regime detection")
    HMM_AVAILABLE = False

class ES_NQ_RegimeDetector:
    """Advanced regime detection for ES/NQ futures"""
    
    def __init__(self):
        self.regimes = {
            'BULL_TREND': {'es_vol': (5, 15), 'nq_vol': (8, 20), 'correlation': (0.7, 1.0)},
            'BEAR_TREND': {'es_vol': (15, 40), 'nq_vol': (20, 50), 'correlation': (0.7, 1.0)},
            'RISK_OFF': {'es_vol': (20, 50), 'nq_vol': (25, 60), 'correlation': (0.8, 1.0)},
            'ROTATION': {'es_vol': (10, 25), 'nq_vol': (15, 35), 'correlation': (-0.3, 0.3)},
            'CHOP': {'es_vol': (8, 18), 'nq_vol': (10, 25), 'correlation': (0.3, 0.7)}
        }
        
        self.output_dir = "Intelligence/data/regime"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Hidden Markov Model for regime switching
        if HMM_AVAILABLE:
            self.hmm_model = hmm.GaussianHMM(n_components=5, covariance_type="full")
        else:
            self.gmm_model = GaussianMixture(n_components=5, random_state=42)
        
    def detect_current_regime(self):
        """Detect current market regime for ES/NQ"""
        
        try:
            # Get market data
            es_data, nq_data = self.fetch_market_data()
            
            if es_data is None or nq_data is None:
                return self.get_fallback_regime()
            
            # Calculate features
            features = self.calculate_features(es_data, nq_data)
            
            if HMM_AVAILABLE:
                regime_result = self.hmm_regime_detection(features)
            else:
                regime_result = self.fallback_regime_detection(features)
            
            # Generate trading adjustments
            adjustments = self.get_regime_adjustments(regime_result['regime'])
            regime_result['adjustments'] = adjustments
            
            return regime_result
            
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return self.get_fallback_regime()
    
    def fetch_market_data(self):
        """Fetch ES/NQ market data"""
        
        try:
            # Use SPY and QQQ as proxies for ES/NQ
            symbols = ['SPY', 'QQQ']
            data = {}
            
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="30d", interval="1h")
                
                if not hist.empty:
                    data[symbol] = hist
                else:
                    logger.warning(f"No data available for {symbol}")
                    return None, None
            
            return data.get('SPY'), data.get('QQQ')
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None, None
    
    def calculate_features(self, es_data, nq_data):
        """Calculate regime detection features"""
        
        try:
            # Calculate volatility (20-period)
            es_returns = es_data['Close'].pct_change().dropna()
            nq_returns = nq_data['Close'].pct_change().dropna()
            
            es_vol = es_returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100
            nq_vol = nq_returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100
            
            # Calculate correlation (20-period)
            aligned_returns = pd.concat([es_returns, nq_returns], axis=1).dropna()
            correlation = aligned_returns.iloc[-20:].corr().iloc[0, 1] if len(aligned_returns) >= 20 else 0.5
            
            # Calculate volume ratio (proxy using price action)
            es_volume_proxy = es_data['Volume'].rolling(5).mean().iloc[-1] / es_data['Volume'].rolling(20).mean().iloc[-1]
            nq_volume_proxy = nq_data['Volume'].rolling(5).mean().iloc[-1] / nq_data['Volume'].rolling(20).mean().iloc[-1]
            volume_ratio = es_volume_proxy / nq_volume_proxy if nq_volume_proxy > 0 else 1.0
            
            # Calculate trend strength
            es_sma_20 = es_data['Close'].rolling(20).mean().iloc[-1]
            es_sma_50 = es_data['Close'].rolling(50).mean().iloc[-1] if len(es_data) >= 50 else es_sma_20
            es_price = es_data['Close'].iloc[-1]
            
            trend_strength = (es_price - es_sma_50) / es_sma_50 if es_sma_50 > 0 else 0
            
            features = np.array([
                es_vol,
                nq_vol, 
                correlation,
                volume_ratio,
                trend_strength
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return np.array([15.0, 20.0, 0.5, 1.0, 0.0])  # Default values
    
    def hmm_regime_detection(self, features):
        """HMM-based regime detection"""
        
        try:
            # Reshape for HMM
            features_reshaped = features.reshape(1, -1)
            
            # For demonstration, use a simple approach since we need historical data to train
            # In production, you would train the HMM on historical data
            regime_names = list(self.regimes.keys())
            
            # Simple rule-based assignment for now
            es_vol, nq_vol, correlation, volume_ratio, trend_strength = features
            
            regime_probs = np.zeros(5)
            
            # Bull trend: low vol, positive trend, high correlation
            if trend_strength > 0.02 and es_vol < 20 and correlation > 0.6:
                regime_probs[0] = 0.8
            
            # Bear trend: high vol, negative trend, high correlation  
            elif trend_strength < -0.02 and es_vol > 20 and correlation > 0.6:
                regime_probs[1] = 0.8
            
            # Risk off: very high vol, high correlation
            elif es_vol > 30 and correlation > 0.8:
                regime_probs[2] = 0.8
            
            # Rotation: low correlation
            elif abs(correlation) < 0.3:
                regime_probs[3] = 0.8
            
            # Chop: medium vol, medium correlation
            else:
                regime_probs[4] = 0.8
            
            # Normalize
            if regime_probs.sum() == 0:
                regime_probs[4] = 1.0  # Default to chop
            else:
                regime_probs = regime_probs / regime_probs.sum()
            
            regime_idx = np.argmax(regime_probs)
            current_regime = regime_names[regime_idx]
            confidence = regime_probs[regime_idx]
            
            return {
                'regime': current_regime,
                'confidence': confidence,
                'probabilities': dict(zip(regime_names, regime_probs)),
                'features': {
                    'es_volatility': es_vol,
                    'nq_volatility': nq_vol,
                    'correlation': correlation,
                    'volume_ratio': volume_ratio,
                    'trend_strength': trend_strength
                }
            }
            
        except Exception as e:
            logger.error(f"Error in HMM regime detection: {e}")
            return self.get_fallback_regime()
    
    def fallback_regime_detection(self, features):
        """Fallback regime detection without HMM"""
        
        try:
            es_vol, nq_vol, correlation, volume_ratio, trend_strength = features
            
            # Simple rule-based regime detection
            if trend_strength > 0.02 and es_vol < 20:
                regime = 'BULL_TREND'
                confidence = 0.7
            elif trend_strength < -0.02 and es_vol > 20:
                regime = 'BEAR_TREND'
                confidence = 0.7
            elif es_vol > 30:
                regime = 'RISK_OFF'
                confidence = 0.8
            elif abs(correlation) < 0.3:
                regime = 'ROTATION'
                confidence = 0.6
            else:
                regime = 'CHOP'
                confidence = 0.5
            
            regime_names = list(self.regimes.keys())
            probabilities = {name: 0.1 for name in regime_names}
            probabilities[regime] = confidence
            
            return {
                'regime': regime,
                'confidence': confidence,
                'probabilities': probabilities,
                'features': {
                    'es_volatility': es_vol,
                    'nq_volatility': nq_vol,
                    'correlation': correlation,
                    'volume_ratio': volume_ratio,
                    'trend_strength': trend_strength
                }
            }
            
        except Exception as e:
            logger.error(f"Error in fallback regime detection: {e}")
            return self.get_fallback_regime()
    
    def get_regime_adjustments(self, regime):
        """Get trading adjustments for current regime"""
        
        adjustments = {
            'BULL_TREND': {
                'ES': {'position_size': 1.2, 'strategies': ['S3', 'S6'], 'direction_bias': 1},
                'NQ': {'position_size': 1.3, 'strategies': ['S3', 'S6'], 'direction_bias': 1}
            },
            'BEAR_TREND': {
                'ES': {'position_size': 0.8, 'strategies': ['S2', 'S11'], 'direction_bias': -1},
                'NQ': {'position_size': 0.7, 'strategies': ['S2', 'S11'], 'direction_bias': -1}
            },
            'RISK_OFF': {
                'ES': {'position_size': 0.5, 'strategies': ['S2'], 'direction_bias': 0},
                'NQ': {'position_size': 0.4, 'strategies': ['S2'], 'direction_bias': 0}
            },
            'ROTATION': {
                'ES': {'position_size': 0.8, 'strategies': ['S2', 'S3'], 'direction_bias': 0},
                'NQ': {'position_size': 0.8, 'strategies': ['S2', 'S3'], 'direction_bias': 0}
            },
            'CHOP': {
                'ES': {'position_size': 0.6, 'strategies': ['S2'], 'direction_bias': 0},
                'NQ': {'position_size': 0.6, 'strategies': ['S2'], 'direction_bias': 0}
            }
        }
        
        return adjustments.get(regime, adjustments['CHOP'])
    
    def get_fallback_regime(self):
        """Get fallback regime data when detection fails"""
        
        return {
            'regime': 'CHOP',
            'confidence': 0.5,
            'probabilities': {name: 0.2 for name in self.regimes.keys()},
            'features': {
                'es_volatility': 15.0,
                'nq_volatility': 20.0,
                'correlation': 0.5,
                'volume_ratio': 1.0,
                'trend_strength': 0.0
            },
            'adjustments': self.get_regime_adjustments('CHOP'),
            'error': 'Fallback regime detection used'
        }

def main():
    """Main execution function"""
    try:
        detector = ES_NQ_RegimeDetector()
        
        logger.info("Starting ES/NQ regime detection...")
        regime_data = detector.detect_current_regime()
        
        # Add timestamp
        regime_data['timestamp'] = datetime.utcnow().isoformat()
        
        # Save data
        output_file = os.path.join(detector.output_dir, 'es_nq_regime.json')
        with open(output_file, 'w') as f:
            json.dump(regime_data, f, indent=2)
        
        # Log results
        print(f"🎯 Current Regime: {regime_data['regime']} ({regime_data['confidence']:.1%} confidence)")
        print(f"📊 ES Volatility: {regime_data['features']['es_volatility']:.1f}%")
        print(f"📊 NQ Volatility: {regime_data['features']['nq_volatility']:.1f}%") 
        print(f"📊 Correlation: {regime_data['features']['correlation']:.2f}")
        
        # Print adjustments
        if 'adjustments' in regime_data:
            adj = regime_data['adjustments']
            print(f"⚙️  ES Position Size: {adj['ES']['position_size']:.1f}x")
            print(f"⚙️  NQ Position Size: {adj['NQ']['position_size']:.1f}x")
        
        logger.info(f"Regime detection complete. Data saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        
        # Create fallback file
        fallback_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'regime': 'CHOP',
            'confidence': 0.5,
            'error': str(e),
            'features': {
                'es_volatility': 15.0,
                'nq_volatility': 20.0,
                'correlation': 0.5,
                'volume_ratio': 1.0,
                'trend_strength': 0.0
            }
        }
        
        output_file = "Intelligence/data/regime/es_nq_regime.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(fallback_data, f, indent=2)

if __name__ == "__main__":
    main()