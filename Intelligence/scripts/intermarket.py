#!/usr/bin/env python3
"""
Intermarket Analysis for ES/NQ Trading Intelligence
Analyzes bonds, commodities, FX correlations for market signals
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import logging
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntermarketAnalyzer:
    def __init__(self):
        self.data_dir = "Intelligence/data/intermarket"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Key intermarket relationships for ES/NQ
        self.symbols = {
            'bonds': ['^TNX', '^TYX'],  # 10yr, 30yr Treasury
            'dollar': ['DX-Y.NYB'],    # Dollar Index
            'commodities': ['GC=F', 'CL=F'],  # Gold, Oil
            'vix': ['^VIX'],
            'crypto': ['BTC-USD'],
            'currencies': ['EURUSD=X', 'GBPUSD=X']
        }
    
    def fetch_intermarket_data(self):
        """Fetch intermarket data for correlation analysis"""
        try:
            data = {}
            
            for category, symbols in self.symbols.items():
                category_data = {}
                for symbol in symbols:
                    try:
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(period="5d", interval="1h")
                        if not hist.empty:
                            # Calculate key metrics
                            current_price = hist['Close'].iloc[-1]
                            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                            change_pct = ((current_price - prev_close) / prev_close) * 100
                            
                            category_data[symbol] = {
                                'current_price': float(current_price),
                                'change_pct': float(change_pct),
                                'volume': float(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0
                            }
                    except Exception as e:
                        logger.warning(f"Failed to fetch {symbol}: {e}")
                
                data[category] = category_data
            
            logger.info("Fetched intermarket data successfully")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching intermarket data: {e}")
            return {}
    
    def analyze_intermarket_signals(self, data):
        """Analyze intermarket relationships for trading signals"""
        try:
            if not data:
                return {}
            
            signals = {}
            
            # Bond analysis (inverse relationship with equities)
            bonds = data.get('bonds', {})
            if bonds:
                tnx_change = bonds.get('^TNX', {}).get('change_pct', 0)
                if tnx_change > 2:  # Rising rates = bearish equities
                    signals['bonds'] = {'signal': 'BEARISH', 'strength': min(0.8, abs(tnx_change) / 5)}
                elif tnx_change < -2:  # Falling rates = bullish equities
                    signals['bonds'] = {'signal': 'BULLISH', 'strength': min(0.8, abs(tnx_change) / 5)}
                else:
                    signals['bonds'] = {'signal': 'NEUTRAL', 'strength': 0.3}
            
            # Dollar analysis (inverse with risk assets)
            dollar = data.get('dollar', {})
            if dollar:
                dxy_change = dollar.get('DX-Y.NYB', {}).get('change_pct', 0)
                if dxy_change > 1:  # Strong dollar = bearish equities
                    signals['dollar'] = {'signal': 'BEARISH', 'strength': min(0.7, abs(dxy_change) / 2)}
                elif dxy_change < -1:  # Weak dollar = bullish equities
                    signals['dollar'] = {'signal': 'BULLISH', 'strength': min(0.7, abs(dxy_change) / 2)}
                else:
                    signals['dollar'] = {'signal': 'NEUTRAL', 'strength': 0.3}
            
            # VIX analysis (inverse with equities)
            vix = data.get('vix', {})
            if vix:
                vix_change = vix.get('^VIX', {}).get('change_pct', 0)
                if vix_change > 10:  # Rising fear = bearish equities
                    signals['vix'] = {'signal': 'BEARISH', 'strength': min(0.9, abs(vix_change) / 20)}
                elif vix_change < -10:  # Falling fear = bullish equities
                    signals['vix'] = {'signal': 'BULLISH', 'strength': min(0.9, abs(vix_change) / 20)}
                else:
                    signals['vix'] = {'signal': 'NEUTRAL', 'strength': 0.3}
            
            # Commodities analysis
            commodities = data.get('commodities', {})
            if commodities:
                gold_change = commodities.get('GC=F', {}).get('change_pct', 0)
                oil_change = commodities.get('CL=F', {}).get('change_pct', 0)
                
                # Gold as safe haven
                if gold_change > 2:  # Rising gold = risk off
                    gold_signal = 'BEARISH'
                elif gold_change < -2:  # Falling gold = risk on
                    gold_signal = 'BULLISH'
                else:
                    gold_signal = 'NEUTRAL'
                
                signals['commodities'] = {
                    'signal': gold_signal,
                    'strength': min(0.6, abs(gold_change) / 4),
                    'gold_change': gold_change,
                    'oil_change': oil_change
                }
            
            # Calculate overall intermarket bias
            bullish_signals = sum(1 for s in signals.values() if s['signal'] == 'BULLISH')
            bearish_signals = sum(1 for s in signals.values() if s['signal'] == 'BEARISH')
            
            # Weight by signal strength
            weighted_bullish = sum(s['strength'] for s in signals.values() if s['signal'] == 'BULLISH')
            weighted_bearish = sum(s['strength'] for s in signals.values() if s['signal'] == 'BEARISH')
            
            net_signal = weighted_bullish - weighted_bearish
            total_signals = len(signals)
            
            if net_signal > 0.5:
                overall_bias = 'BULLISH'
                confidence = min(0.8, net_signal / 2)
            elif net_signal < -0.5:
                overall_bias = 'BEARISH'
                confidence = min(0.8, abs(net_signal) / 2)
            else:
                overall_bias = 'NEUTRAL'
                confidence = 0.4
            
            analysis = {
                'intermarket_bias': overall_bias,
                'signal_strength': confidence,
                'individual_signals': signals,
                'net_signal_score': net_signal,
                'bullish_signals': bullish_signals,
                'bearish_signals': bearish_signals,
                'trade_recommendation': {
                    'direction': overall_bias,
                    'confidence': confidence,
                    'timeframe': 'MEDIUM_TERM',  # Intermarket moves are slower
                    'position_size_multiplier': 1.0 + (confidence * 0.3)
                }
            }
            
            logger.info(f"Intermarket Analysis: {overall_bias} bias with {confidence:.2f} confidence")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing intermarket signals: {e}")
            return {}
    
    def save_analysis(self, analysis):
        """Save intermarket analysis to file"""
        try:
            output_file = os.path.join(self.data_dir, 'latest_intermarket_analysis.json')
            
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            logger.info(f"Saved intermarket analysis to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving intermarket analysis: {e}")

def main():
    """Main intermarket analysis workflow"""
    logger.info("Starting Intermarket analysis...")
    
    analyzer = IntermarketAnalyzer()
    
    # Fetch and analyze intermarket data
    data = analyzer.fetch_intermarket_data()
    if data:
        analysis = analyzer.analyze_intermarket_signals(data)
        if analysis:
            analyzer.save_analysis(analysis)
            logger.info("Intermarket analysis completed successfully")
        else:
            logger.error("Failed to analyze intermarket data")
    else:
        logger.error("Failed to fetch intermarket data")

if __name__ == "__main__":
    main()
