#!/usr/bin/env python3
"""
Microstructure Analysis for Intelligence Pipeline
Analyzes order flow patterns, delta divergence, and absorption
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

class MicrostructureAnalyzer:
    def __init__(self):
        self.data_dir = "Intelligence/data/microstructure"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def analyze_order_flow(self, symbol='ES=F'):
        """Analyze order flow patterns"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="2d", interval="1m")
            
            if data.empty:
                return {}
            
            # Calculate VWAP
            data['VWAP'] = ta.volume.volume_weighted_average_price(
                data['High'], data['Low'], data['Close'], data['Volume'])
            
            # Delta approximation (simplified)
            data['Delta'] = np.where(data['Close'] > data['Open'], 
                                   data['Volume'], -data['Volume'])
            data['Cumulative_Delta'] = data['Delta'].cumsum()
            
            # VWAP deviation
            current_price = data['Close'].iloc[-1]
            current_vwap = data['VWAP'].iloc[-1]
            vwap_deviation = ((current_price - current_vwap) / current_vwap) * 100
            
            # Volume analysis
            avg_volume = data['Volume'].tail(20).mean()
            current_volume = data['Volume'].tail(5).mean()
            volume_surge = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Delta divergence detection
            price_change = (data['Close'].iloc[-1] - data['Close'].iloc[-20]) / data['Close'].iloc[-20]
            delta_change = (data['Cumulative_Delta'].iloc[-1] - data['Cumulative_Delta'].iloc[-20])
            
            analysis = {
                'vwap_deviation': round(vwap_deviation, 3),
                'volume_surge': round(volume_surge, 2),
                'delta_divergence': abs(price_change) > 0.001 and delta_change * price_change < 0,
                'absorption_detected': volume_surge > 2.0 and abs(vwap_deviation) < 0.1,
                'current_price': round(current_price, 2),
                'vwap': round(current_vwap, 2)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing microstructure: {e}")
            return {}

def main():
    """Main execution function"""
    logger.info("Starting microstructure analysis...")
    
    analyzer = MicrostructureAnalyzer()
    analysis = analyzer.analyze_order_flow()
    
    # Save results
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M")
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'microstructure_analysis': analysis,
        'generated_by': 'microstructure_analyzer'
    }
    
    filename = f"{timestamp_str}.json"
    filepath = os.path.join(analyzer.data_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info("Microstructure analysis completed")

if __name__ == "__main__":
    main()