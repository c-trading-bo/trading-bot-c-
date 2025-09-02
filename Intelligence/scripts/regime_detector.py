#!/usr/bin/env python3
"""
Regime Detector for Intelligence Pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import logging
import yfinance as yf
import ta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RegimeDetector:
    def __init__(self):
        self.data_dir = "Intelligence/data/regime"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def detect_market_regime(self):
        """Detect current market regime"""
        try:
            # Get SPY data
            spy = yf.Ticker('SPY')
            data = spy.history(period="60d")
            
            if data.empty:
                return {}
            
            # Calculate ADX for trend strength
            data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'])
            
            # Calculate ATR for volatility
            data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
            
            # Moving average alignment
            data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
            data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
            
            current_adx = data['ADX'].iloc[-1] if not data['ADX'].isna().all() else 25
            current_atr = data['ATR'].iloc[-1] if not data['ATR'].isna().all() else 1
            price = data['Close'].iloc[-1]
            sma_20 = data['SMA_20'].iloc[-1]
            sma_50 = data['SMA_50'].iloc[-1]
            
            # Determine regime
            if current_adx > 30 and price > sma_20 > sma_50:
                regime = 'trending_up'
                strength = 'strong'
            elif current_adx > 30 and price < sma_20 < sma_50:
                regime = 'trending_down'
                strength = 'strong'
            elif current_adx < 20:
                regime = 'ranging'
                strength = 'weak'
            else:
                regime = 'transitioning'
                strength = 'moderate'
            
            analysis = {
                'regime_type': regime,
                'trend_strength': strength,
                'adx_value': round(current_adx, 1),
                'atr_value': round(current_atr, 2),
                'transition_probability': 0.3 if regime == 'transitioning' else 0.1
            }
            
            return analysis
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return {}

def main():
    detector = RegimeDetector()
    analysis = detector.detect_market_regime()
    
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H")
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'regime_analysis': analysis,
        'generated_by': 'regime_detector'
    }
    
    filepath = os.path.join(detector.data_dir, f"{timestamp_str}.json")
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info("Regime detection completed")

if __name__ == "__main__":
    main()