#!/usr/bin/env python3
"""
Free Options Flow Analyzer
Uses free data sources for basic gamma/vanna analysis
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class FreeOptionsAnalyzer:
    """Analyzes options flow using free data sources"""
    
    def __init__(self):
        self.vix_symbols = ['^VIX', '^VIX9D', '^VIX3M', '^VIX6M']
        
    def get_vix_term_structure(self):
        """Get VIX term structure from Yahoo Finance (FREE)"""
        try:
            vix_data = {}
            for symbol in self.vix_symbols:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="5d", interval="1d")
                if not data.empty:
                    vix_data[symbol.replace('^', '')] = data['Close'].iloc[-1]
            
            return vix_data
        except Exception as e:
            logger.error(f"Error getting VIX data: {e}")
            return {}
    
    def estimate_gamma_exposure(self, symbol='SPY'):
        """Estimate gamma exposure using free options data"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current price
            info = ticker.info
            current_price = info.get('currentPrice', 0)
            
            # Get options expiration dates
            expirations = ticker.options
            if not expirations:
                return {}
            
            # Use nearest expiration
            nearest_exp = expirations[0]
            options_chain = ticker.option_chain(nearest_exp)
            
            calls = options_chain.calls
            puts = options_chain.puts
            
            # Simple gamma exposure estimation
            call_oi = calls['openInterest'].sum()
            put_oi = puts['openInterest'].sum()
            
            # Estimate dealer positioning
            if call_oi > put_oi * 1.5:
                gamma_exposure = 'negative'  # Dealers short gamma
                expected_movement = 'suppressed'
            elif put_oi > call_oi * 1.5:
                gamma_exposure = 'positive'  # Dealers long gamma
                expected_movement = 'amplified'
            else:
                gamma_exposure = 'neutral'
                expected_movement = 'normal'
            
            return {
                'gamma_exposure': gamma_exposure,
                'expected_movement': expected_movement,
                'call_oi': int(call_oi),
                'put_oi': int(put_oi),
                'current_price': current_price,
                'analyzed_symbol': symbol
            }
            
        except Exception as e:
            logger.error(f"Error estimating gamma exposure: {e}")
            return {}
    
    def get_put_call_ratio(self, symbol='SPY'):
        """Calculate put/call ratio from free data"""
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            
            if not expirations:
                return 1.0
            
            total_call_volume = 0
            total_put_volume = 0
            
            # Analyze first 3 expirations
            for exp in expirations[:3]:
                chain = ticker.option_chain(exp)
                total_call_volume += chain.calls['volume'].fillna(0).sum()
                total_put_volume += chain.puts['volume'].fillna(0).sum()
            
            if total_call_volume == 0:
                return 999  # All puts
            
            return total_put_volume / total_call_volume
            
        except Exception as e:
            logger.error(f"Error calculating P/C ratio: {e}")
            return 1.0
    
    def generate_options_signals(self):
        """Generate trading signals from options analysis"""
        try:
            vix_data = self.get_vix_term_structure()
            gamma_data = self.estimate_gamma_exposure('SPY')
            pc_ratio = self.get_put_call_ratio('SPY')
            
            signals = []
            
            # VIX signals
            if 'VIX' in vix_data:
                vix_level = vix_data['VIX']
                
                if vix_level < 15:
                    signals.append({
                        'type': 'volatility_regime',
                        'signal': 'VIX extremely low - expect volatility expansion',
                        'confidence': 'high',
                        'vix_level': vix_level
                    })
                elif vix_level > 30:
                    signals.append({
                        'type': 'volatility_regime', 
                        'signal': 'VIX elevated - expect mean reversion',
                        'confidence': 'medium',
                        'vix_level': vix_level
                    })
            
            # Gamma exposure signals
            if gamma_data:
                if gamma_data['gamma_exposure'] == 'negative':
                    signals.append({
                        'type': 'gamma_exposure',
                        'signal': 'Dealers short gamma - expect range-bound action',
                        'confidence': 'medium'
                    })
                elif gamma_data['gamma_exposure'] == 'positive':
                    signals.append({
                        'type': 'gamma_exposure',
                        'signal': 'Dealers long gamma - expect trending moves',
                        'confidence': 'medium'
                    })
            
            # Put/Call ratio signals
            if pc_ratio > 1.2:
                signals.append({
                    'type': 'sentiment',
                    'signal': 'High put/call ratio - contrarian bullish',
                    'confidence': 'low',
                    'pc_ratio': pc_ratio
                })
            elif pc_ratio < 0.7:
                signals.append({
                    'type': 'sentiment',
                    'signal': 'Low put/call ratio - contrarian bearish', 
                    'confidence': 'low',
                    'pc_ratio': pc_ratio
                })
            
            return {
                'signals': signals,
                'vix_data': vix_data,
                'gamma_analysis': gamma_data,
                'put_call_ratio': pc_ratio,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating options signals: {e}")
            return {'signals': [], 'error': str(e)}

if __name__ == "__main__":
    analyzer = FreeOptionsAnalyzer()
    results = analyzer.generate_options_signals()
    print(f"Options Analysis: {results}")
