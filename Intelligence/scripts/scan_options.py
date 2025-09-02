#!/usr/bin/env python3
"""
Options Flow Scanner for Intelligence Pipeline
Scans SPY/QQQ options for unusual activity, put/call ratios, and max pain analysis
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptionsFlowScanner:
    def __init__(self):
        self.data_dir = "Intelligence/data/options/flow"
        os.makedirs(self.data_dir, exist_ok=True)
        self.symbols = ['SPY', 'QQQ', 'IWM', 'DIA']
        
    def scan_options_activity(self):
        """Scan for unusual options activity"""
        try:
            results = {}
            
            for symbol in self.symbols:
                ticker = yf.Ticker(symbol)
                
                # Get current price
                hist = ticker.history(period="1d")
                current_price = hist['Close'].iloc[-1] if not hist.empty else 0
                
                # Get options expirations
                expirations = ticker.options
                if not expirations:
                    continue
                    
                # Analyze nearest expiration
                nearest_exp = expirations[0] if len(expirations) > 0 else None
                if nearest_exp:
                    calls = ticker.option_chain(nearest_exp).calls
                    puts = ticker.option_chain(nearest_exp).puts
                    
                    # Calculate metrics
                    call_volume = calls['volume'].fillna(0).sum()
                    put_volume = puts['volume'].fillna(0).sum()
                    total_volume = call_volume + put_volume
                    
                    put_call_ratio = put_volume / call_volume if call_volume > 0 else 0
                    
                    # Calculate max pain (simplified)
                    max_pain = self.calculate_max_pain(calls, puts, current_price)
                    
                    # Detect unusual activity
                    unusual_calls = calls[calls['volume'] > calls['volume'].quantile(0.9)]
                    unusual_puts = puts[puts['volume'] > puts['volume'].quantile(0.9)]
                    
                    results[symbol] = {
                        'current_price': round(float(current_price), 2),
                        'call_volume': int(call_volume),
                        'put_volume': int(put_volume),
                        'total_volume': int(total_volume),
                        'put_call_ratio': round(put_call_ratio, 3),
                        'max_pain': round(max_pain, 2),
                        'max_pain_distance': round(abs(current_price - max_pain) / current_price * 100, 2),
                        'unusual_calls_count': len(unusual_calls),
                        'unusual_puts_count': len(unusual_puts),
                        'zero_dte_volume': self.get_zero_dte_volume(calls, puts, nearest_exp)
                    }
                    
            return results
            
        except Exception as e:
            logger.error(f"Error scanning options: {e}")
            return {}
    
    def calculate_max_pain(self, calls, puts, current_price):
        """Calculate max pain level (simplified)"""
        try:
            # Get strikes around current price
            strikes = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
            strikes = [s for s in strikes if abs(s - current_price) / current_price < 0.2]  # Within 20%
            
            if not strikes:
                return current_price
                
            max_pain_strike = strikes[len(strikes)//2]  # Simplified: middle strike
            return max_pain_strike
            
        except Exception:
            return current_price
    
    def get_zero_dte_volume(self, calls, puts, expiration):
        """Check for 0DTE volume"""
        try:
            exp_date = datetime.strptime(expiration, '%Y-%m-%d').date()
            today = datetime.now().date()
            
            if exp_date == today:
                zero_dte_calls = calls['volume'].fillna(0).sum()
                zero_dte_puts = puts['volume'].fillna(0).sum()
                return int(zero_dte_calls + zero_dte_puts)
            return 0
            
        except Exception:
            return 0
    
    def analyze_sentiment(self, options_data):
        """Analyze overall options sentiment"""
        try:
            if not options_data:
                return {'sentiment': 'neutral', 'confidence': 0, 'signals': []}
            
            # Calculate aggregate metrics
            total_call_volume = sum(data['call_volume'] for data in options_data.values())
            total_put_volume = sum(data['put_volume'] for data in options_data.values())
            
            overall_pcr = total_put_volume / total_call_volume if total_call_volume > 0 else 1
            
            # Determine sentiment
            signals = []
            if overall_pcr > 1.2:
                sentiment = 'bearish'
                signals.append('High put/call ratio indicates bearish sentiment')
            elif overall_pcr < 0.7:
                sentiment = 'bullish'
                signals.append('Low put/call ratio indicates bullish sentiment')
            else:
                sentiment = 'neutral'
            
            # Check for unusual activity
            for symbol, data in options_data.items():
                if data['unusual_calls_count'] > 5:
                    signals.append(f'{symbol}: Unusual call activity detected')
                if data['unusual_puts_count'] > 5:
                    signals.append(f'{symbol}: Unusual put activity detected')
                if data['zero_dte_volume'] > 10000:
                    signals.append(f'{symbol}: High 0DTE volume')
            
            confidence = min(1.0, abs(overall_pcr - 1.0) * 2)
            
            return {
                'sentiment': sentiment,
                'confidence': round(confidence, 3),
                'overall_pcr': round(overall_pcr, 3),
                'signals': signals
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'sentiment': 'neutral', 'confidence': 0, 'signals': []}
    
    def save_data(self, options_data, analysis):
        """Save options flow data"""
        try:
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M")
            
            # Combine data
            output_data = {
                'timestamp': datetime.now().isoformat(),
                'options_data': options_data,
                'analysis': analysis,
                'generated_by': 'options_flow_scanner'
            }
            
            # Save timestamped file
            filename = f"{timestamp_str}.json"
            filepath = os.path.join(self.data_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Options flow data saved: {len(options_data)} symbols analyzed, sentiment={analysis['sentiment']}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return None

def main():
    """Main execution function"""
    logger.info("Starting options flow analysis...")
    
    scanner = OptionsFlowScanner()
    
    # Scan options activity
    options_data = scanner.scan_options_activity()
    
    # Analyze sentiment
    analysis = scanner.analyze_sentiment(options_data)
    
    # Save data
    filepath = scanner.save_data(options_data, analysis)
    
    if filepath:
        logger.info("Options flow analysis completed successfully")
    else:
        logger.error("Options flow analysis failed")

if __name__ == "__main__":
    main()