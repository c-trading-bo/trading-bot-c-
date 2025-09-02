#!/usr/bin/env python3
"""
Failed Patterns Detection for Intelligence Pipeline
Tracks failed breakouts for reversal trades
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import logging
import ta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FailedPatternsDetector:
    def __init__(self):
        self.data_dir = "Intelligence/data/patterns/failed"
        os.makedirs(self.data_dir, exist_ok=True)
        self.symbols = ['ES=F', 'NQ=F', 'SPY', 'QQQ']
        
    def detect_failed_breakouts(self, symbol):
        """Detect failed breakout patterns"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="10d", interval="15m")
            
            if data.empty:
                return []
            
            failed_patterns = []
            
            # Calculate technical indicators
            data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
            data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
            data['Volume_SMA'] = ta.volume.volume_sma(data['Close'], data['Volume'], window=20)
            
            # Detect breakout attempts
            for i in range(20, len(data)-5):
                high_20 = data['High'].iloc[i-20:i].max()
                low_20 = data['Low'].iloc[i-20:i].min()
                current_close = data['Close'].iloc[i]
                
                # Upside breakout attempt
                if (current_close > high_20 and 
                    data['Volume'].iloc[i] > data['Volume_SMA'].iloc[i] * 1.5):
                    
                    # Check if it failed (price returned below breakout level within 5 periods)
                    failed = False
                    for j in range(i+1, min(i+6, len(data))):
                        if data['Close'].iloc[j] < high_20 * 0.998:  # Allow small tolerance
                            failed = True
                            break
                    
                    if failed:
                        failed_patterns.append({
                            'type': 'failed_upside_breakout',
                            'timestamp': data.index[i].isoformat(),
                            'breakout_level': round(high_20, 2),
                            'current_price': round(data['Close'].iloc[-1], 2),
                            'volume_surge': round(data['Volume'].iloc[i] / data['Volume_SMA'].iloc[i], 2),
                            'reversal_target': round(low_20, 2),
                            'probability': self.calculate_failure_probability('upside', data.iloc[i])
                        })
                
                # Downside breakout attempt  
                elif (current_close < low_20 and 
                      data['Volume'].iloc[i] > data['Volume_SMA'].iloc[i] * 1.5):
                    
                    failed = False
                    for j in range(i+1, min(i+6, len(data))):
                        if data['Close'].iloc[j] > low_20 * 1.002:
                            failed = True
                            break
                    
                    if failed:
                        failed_patterns.append({
                            'type': 'failed_downside_breakout',
                            'timestamp': data.index[i].isoformat(),
                            'breakout_level': round(low_20, 2),
                            'current_price': round(data['Close'].iloc[-1], 2),
                            'volume_surge': round(data['Volume'].iloc[i] / data['Volume_SMA'].iloc[i], 2),
                            'reversal_target': round(high_20, 2),
                            'probability': self.calculate_failure_probability('downside', data.iloc[i])
                        })
            
            return failed_patterns[-10:]  # Return most recent
            
        except Exception as e:
            logger.error(f"Error detecting failed patterns for {symbol}: {e}")
            return []
    
    def calculate_failure_probability(self, direction, bar_data):
        """Calculate probability of pattern failure success"""
        # Simplified probability based on RSI and volume
        rsi = bar_data.get('RSI', 50)
        
        if direction == 'upside':
            # Failed upside breakout more likely to reverse when RSI was high
            prob = min(0.9, max(0.3, (rsi - 50) / 50))
        else:
            # Failed downside breakout more likely to reverse when RSI was low
            prob = min(0.9, max(0.3, (50 - rsi) / 50))
        
        return round(prob, 3)
    
    def generate_reversal_signals(self, all_patterns):
        """Generate reversal signals from failed patterns"""
        signals = []
        
        for symbol, patterns in all_patterns.items():
            for pattern in patterns:
                # Recent failed patterns (within last 2 hours)
                pattern_time = datetime.fromisoformat(pattern['timestamp'].replace('Z', '+00:00'))
                if (datetime.now() - pattern_time.replace(tzinfo=None)).total_seconds() < 7200:
                    
                    signal = {
                        'symbol': symbol,
                        'type': 'failed_pattern_reversal',
                        'direction': 'short' if 'upside' in pattern['type'] else 'long',
                        'entry_level': pattern['current_price'],
                        'target': pattern['reversal_target'],
                        'pattern_type': pattern['type'],
                        'probability': pattern['probability'],
                        'confidence': 'high' if pattern['probability'] > 0.7 else 'medium'
                    }
                    signals.append(signal)
        
        return signals

def main():
    """Main execution function"""
    logger.info("Starting failed patterns detection...")
    
    detector = FailedPatternsDetector()
    all_patterns = {}
    
    for symbol in detector.symbols:
        patterns = detector.detect_failed_breakouts(symbol)
        if patterns:
            all_patterns[symbol] = patterns
    
    # Generate signals
    signals = detector.generate_reversal_signals(all_patterns)
    
    # Save results
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M")
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'failed_patterns': all_patterns,
        'reversal_signals': signals,
        'summary': {
            'symbols_analyzed': len(detector.symbols),
            'patterns_detected': sum(len(p) for p in all_patterns.values()),
            'signals_generated': len(signals)
        },
        'generated_by': 'failed_patterns_detector'
    }
    
    filename = f"{timestamp_str}.json"
    filepath = os.path.join(detector.data_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Failed patterns analysis completed: {len(all_patterns)} symbols, {sum(len(p) for p in all_patterns.values())} patterns")

if __name__ == "__main__":
    main()