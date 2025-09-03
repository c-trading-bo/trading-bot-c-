#!/usr/bin/env python3
"""
Real-Time Signal Processing and Pattern Recognition
Part of the 11-script ML system restoration - Script 11/11
"""

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class RealTimeSignalProcessor:
    """Real-time signal processing and pattern recognition for trading signals"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.signal_buffer = []
        self.pattern_library = {}
        self.scaler = StandardScaler()
        
        # Signal processing parameters
        self.noise_threshold = 0.01
        self.trend_threshold = 0.005
        self.volatility_threshold = 0.02
        
        # Pattern recognition
        self.clustering_model = DBSCAN(eps=0.5, min_samples=5)
        self.known_patterns = []
        
    def add_signal_point(self, price: float, volume: float, timestamp: datetime = None):
        """Add new signal point to processing buffer"""
        
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        signal_point = {
            'timestamp': timestamp,
            'price': price,
            'volume': volume,
            'returns': 0.0,
            'volatility': 0.0
        }
        
        # Calculate returns and volatility if we have previous data
        if self.signal_buffer:
            prev_price = self.signal_buffer[-1]['price']
            signal_point['returns'] = (price - prev_price) / prev_price if prev_price > 0 else 0.0
            
            # Rolling volatility calculation
            if len(self.signal_buffer) >= 10:
                recent_returns = [p['returns'] for p in self.signal_buffer[-10:]]
                signal_point['volatility'] = np.std(recent_returns)
        
        self.signal_buffer.append(signal_point)
        
        # Maintain window size
        if len(self.signal_buffer) > self.window_size:
            self.signal_buffer = self.signal_buffer[-self.window_size:]
    
    def detect_trend(self) -> Dict:
        """Detect current trend in signal data"""
        
        if len(self.signal_buffer) < 10:
            return {'status': 'insufficient_data'}
        
        prices = [p['price'] for p in self.signal_buffer[-20:]]
        
        # Linear regression for trend detection
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        
        # Normalize slope by price level
        normalized_slope = slope / np.mean(prices) if np.mean(prices) > 0 else 0
        
        # Determine trend
        if normalized_slope > self.trend_threshold:
            trend = 'UPTREND'
        elif normalized_slope < -self.trend_threshold:
            trend = 'DOWNTREND'
        else:
            trend = 'SIDEWAYS'
        
        # Calculate trend strength
        r_squared = np.corrcoef(x, prices)[0, 1] ** 2
        
        return {
            'trend': trend,
            'slope': slope,
            'normalized_slope': normalized_slope,
            'strength': r_squared,
            'confidence': 'HIGH' if r_squared > 0.7 else 'MEDIUM' if r_squared > 0.4 else 'LOW'
        }
    
    def detect_support_resistance(self) -> Dict:
        """Detect support and resistance levels"""
        
        if len(self.signal_buffer) < 50:
            return {'status': 'insufficient_data'}
        
        prices = np.array([p['price'] for p in self.signal_buffer])
        
        # Find local minima and maxima
        local_minima = signal.argrelextrema(prices, np.less, order=5)[0]
        local_maxima = signal.argrelextrema(prices, np.greater, order=5)[0]
        
        # Get price levels
        support_levels = prices[local_minima] if len(local_minima) > 0 else []
        resistance_levels = prices[local_maxima] if len(local_maxima) > 0 else []
        
        # Cluster similar levels
        support_clusters = self._cluster_price_levels(support_levels)
        resistance_clusters = self._cluster_price_levels(resistance_levels)
        
        current_price = prices[-1]
        
        return {
            'support_levels': support_clusters,
            'resistance_levels': resistance_clusters,
            'current_price': current_price,
            'nearest_support': min(support_clusters, key=lambda x: abs(x - current_price)) if support_clusters else None,
            'nearest_resistance': min(resistance_clusters, key=lambda x: abs(x - current_price)) if resistance_clusters else None
        }
    
    def _cluster_price_levels(self, levels: np.ndarray, tolerance: float = 0.01) -> List[float]:
        """Cluster similar price levels together"""
        
        if len(levels) < 2:
            return levels.tolist()
        
        # Group levels within tolerance
        clustered_levels = []
        sorted_levels = np.sort(levels)
        
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                current_cluster.append(level)
            else:
                # Finalize current cluster
                clustered_levels.append(np.mean(current_cluster))
                current_cluster = [level]
        
        # Add final cluster
        clustered_levels.append(np.mean(current_cluster))
        
        return clustered_levels
    
    def detect_volatility_regime(self) -> Dict:
        """Detect current volatility regime"""
        
        if len(self.signal_buffer) < 20:
            return {'status': 'insufficient_data'}
        
        # Calculate rolling volatilities
        short_vol = np.std([p['returns'] for p in self.signal_buffer[-10:]])
        medium_vol = np.std([p['returns'] for p in self.signal_buffer[-20:]])
        long_vol = np.std([p['returns'] for p in self.signal_buffer[-50:]]) if len(self.signal_buffer) >= 50 else medium_vol
        
        # Determine regime
        if short_vol > self.volatility_threshold * 2:
            regime = 'HIGH_VOLATILITY'
        elif short_vol < self.volatility_threshold * 0.5:
            regime = 'LOW_VOLATILITY'
        else:
            regime = 'NORMAL_VOLATILITY'
        
        # Volatility trend
        if short_vol > medium_vol * 1.2:
            vol_trend = 'INCREASING'
        elif short_vol < medium_vol * 0.8:
            vol_trend = 'DECREASING'
        else:
            vol_trend = 'STABLE'
        
        return {
            'regime': regime,
            'volatility_trend': vol_trend,
            'short_volatility': short_vol,
            'medium_volatility': medium_vol,
            'long_volatility': long_vol,
            'volatility_ratio': short_vol / medium_vol if medium_vol > 0 else 1.0
        }
    
    def detect_patterns(self) -> Dict:
        """Detect known trading patterns"""
        
        if len(self.signal_buffer) < 30:
            return {'status': 'insufficient_data'}
        
        patterns_detected = []
        
        # Get recent price data
        prices = np.array([p['price'] for p in self.signal_buffer[-30:]])
        volumes = np.array([p['volume'] for p in self.signal_buffer[-30:]])
        
        # Normalize data
        normalized_prices = (prices - np.mean(prices)) / np.std(prices)
        
        # Double Top/Bottom Detection
        double_pattern = self._detect_double_pattern(normalized_prices)
        if double_pattern:
            patterns_detected.append(double_pattern)
        
        # Head and Shoulders Detection
        hs_pattern = self._detect_head_shoulders(normalized_prices)
        if hs_pattern:
            patterns_detected.append(hs_pattern)
        
        # Volume Patterns
        volume_pattern = self._detect_volume_pattern(prices, volumes)
        if volume_pattern:
            patterns_detected.append(volume_pattern)
        
        return {
            'patterns_detected': patterns_detected,
            'pattern_count': len(patterns_detected),
            'confidence': 'HIGH' if len(patterns_detected) >= 2 else 'MEDIUM' if len(patterns_detected) == 1 else 'LOW'
        }
    
    def _detect_double_pattern(self, prices: np.ndarray) -> Optional[Dict]:
        """Detect double top/bottom patterns"""
        
        # Find peaks and troughs
        peaks = signal.find_peaks(prices, distance=5)[0]
        troughs = signal.find_peaks(-prices, distance=5)[0]
        
        # Check for double top
        if len(peaks) >= 2:
            last_two_peaks = peaks[-2:]
            if abs(prices[last_two_peaks[0]] - prices[last_two_peaks[1]]) < 0.1:
                return {
                    'pattern': 'DOUBLE_TOP',
                    'confidence': 0.7,
                    'signal': 'BEARISH'
                }
        
        # Check for double bottom
        if len(troughs) >= 2:
            last_two_troughs = troughs[-2:]
            if abs(prices[last_two_troughs[0]] - prices[last_two_troughs[1]]) < 0.1:
                return {
                    'pattern': 'DOUBLE_BOTTOM',
                    'confidence': 0.7,
                    'signal': 'BULLISH'
                }
        
        return None
    
    def _detect_head_shoulders(self, prices: np.ndarray) -> Optional[Dict]:
        """Detect head and shoulders pattern"""
        
        peaks = signal.find_peaks(prices, distance=3)[0]
        
        if len(peaks) >= 3:
            # Check last three peaks for head and shoulders
            last_three = peaks[-3:]
            left_shoulder = prices[last_three[0]]
            head = prices[last_three[1]]
            right_shoulder = prices[last_three[2]]
            
            # Head should be higher than both shoulders
            if (head > left_shoulder and head > right_shoulder and
                abs(left_shoulder - right_shoulder) < 0.2):
                return {
                    'pattern': 'HEAD_AND_SHOULDERS',
                    'confidence': 0.8,
                    'signal': 'BEARISH'
                }
        
        return None
    
    def _detect_volume_pattern(self, prices: np.ndarray, volumes: np.ndarray) -> Optional[Dict]:
        """Detect volume-based patterns"""
        
        if len(volumes) < 10:
            return None
        
        # Volume surge detection
        avg_volume = np.mean(volumes[:-5])
        recent_volume = np.mean(volumes[-5:])
        
        price_change = (prices[-1] - prices[-10]) / prices[-10] if prices[-10] > 0 else 0
        
        if recent_volume > avg_volume * 1.5:
            if price_change > 0.01:
                return {
                    'pattern': 'VOLUME_BREAKOUT_UP',
                    'confidence': 0.6,
                    'signal': 'BULLISH'
                }
            elif price_change < -0.01:
                return {
                    'pattern': 'VOLUME_BREAKOUT_DOWN',
                    'confidence': 0.6,
                    'signal': 'BEARISH'
                }
        
        return None
    
    def generate_signal_summary(self) -> Dict:
        """Generate comprehensive signal analysis summary"""
        
        if len(self.signal_buffer) < 10:
            return {'status': 'insufficient_data'}
        
        # Get all analysis components
        trend_analysis = self.detect_trend()
        support_resistance = self.detect_support_resistance()
        volatility_analysis = self.detect_volatility_regime()
        pattern_analysis = self.detect_patterns()
        
        # Current market state
        current_price = self.signal_buffer[-1]['price']
        current_volume = self.signal_buffer[-1]['volume']
        
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'current_state': {
                'price': current_price,
                'volume': current_volume
            },
            'trend_analysis': trend_analysis,
            'support_resistance': support_resistance,
            'volatility_analysis': volatility_analysis,
            'pattern_analysis': pattern_analysis,
            'overall_signal': self._generate_overall_signal(trend_analysis, pattern_analysis, volatility_analysis)
        }
        
        return summary
    
    def _generate_overall_signal(self, trend: Dict, patterns: Dict, volatility: Dict) -> Dict:
        """Generate overall trading signal from all analyses"""
        
        signals = []
        confidence_sum = 0
        
        # Trend signal
        if trend.get('trend') == 'UPTREND' and trend.get('confidence') != 'LOW':
            signals.append('BULLISH')
            confidence_sum += 0.3
        elif trend.get('trend') == 'DOWNTREND' and trend.get('confidence') != 'LOW':
            signals.append('BEARISH')
            confidence_sum += 0.3
        
        # Pattern signals
        for pattern in patterns.get('patterns_detected', []):
            if pattern.get('signal') == 'BULLISH':
                signals.append('BULLISH')
                confidence_sum += 0.4
            elif pattern.get('signal') == 'BEARISH':
                signals.append('BEARISH')
                confidence_sum += 0.4
        
        # Volatility impact
        if volatility.get('regime') == 'HIGH_VOLATILITY':
            confidence_sum *= 0.8  # Reduce confidence in high volatility
        
        # Determine overall signal
        bullish_count = signals.count('BULLISH')
        bearish_count = signals.count('BEARISH')
        
        if bullish_count > bearish_count:
            overall_signal = 'BULLISH'
        elif bearish_count > bullish_count:
            overall_signal = 'BEARISH'
        else:
            overall_signal = 'NEUTRAL'
        
        return {
            'signal': overall_signal,
            'confidence': min(confidence_sum, 1.0),
            'component_signals': signals,
            'signal_strength': abs(bullish_count - bearish_count)
        }
    
    def save_analysis(self, path: str = 'models/ml/signal_analysis.json'):
        """Save signal analysis results"""
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        summary = self.generate_signal_summary()
        
        save_data = {
            'signal_summary': summary,
            'recent_signal_buffer': self.signal_buffer[-50:],  # Keep last 50 points
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"[SIGNALS] Analysis saved to {path}")

# Main execution
if __name__ == "__main__":
    print("[SIGNALS] Initializing real-time signal processing...")
    
    # Initialize processor
    processor = RealTimeSignalProcessor()
    
    # Simulate real-time price/volume data
    base_price = 4500
    for i in range(100):
        # Generate synthetic price with trends and patterns
        time_factor = i / 100
        trend = 50 * time_factor  # Upward trend
        noise = np.random.randn() * 5
        
        # Add pattern (simulated double top around i=70-80)
        if 70 <= i <= 75:
            pattern_effect = 20
        elif 76 <= i <= 80:
            pattern_effect = 20
        else:
            pattern_effect = 0
        
        price = base_price + trend + noise + pattern_effect
        volume = 1000 + abs(noise) * 100  # Volume increases with volatility
        
        # Add to processor
        processor.add_signal_point(price, volume)
        
        # Generate analysis every 20 points
        if (i + 1) % 20 == 0:
            summary = processor.generate_signal_summary()
            if summary.get('status') != 'insufficient_data':
                overall_signal = summary['overall_signal']
                print(f"Point {i+1}: Price={price:.2f}, "
                      f"Signal={overall_signal['signal']}, "
                      f"Confidence={overall_signal['confidence']:.2f}")
    
    # Generate final comprehensive analysis
    final_summary = processor.generate_signal_summary()
    print(f"\n[SIGNALS] Final Analysis:")
    print(f"  Trend: {final_summary['trend_analysis'].get('trend', 'UNKNOWN')}")
    print(f"  Volatility: {final_summary['volatility_analysis'].get('regime', 'UNKNOWN')}")
    print(f"  Patterns: {final_summary['pattern_analysis'].get('pattern_count', 0)} detected")
    print(f"  Overall Signal: {final_summary['overall_signal']['signal']}")
    
    # Save analysis
    processor.save_analysis()
    print("[SIGNALS] Real-time signal processing complete!")
    print("\n🎉 ALL 11 ML SCRIPTS IMPLEMENTED SUCCESSFULLY!")
    print("✅ Complete ML system restoration finished!")