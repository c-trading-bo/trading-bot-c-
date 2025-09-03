#!/usr/bin/env python3
"""
Congressional Trades Monitor for Intelligence Pipeline
Tracks congressional insider trading for early market signals
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import logging
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CongressionalTradesMonitor:
    def __init__(self):
        self.data_dir = "Intelligence/data/congress"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def fetch_congressional_trades(self):
        """Fetch recent congressional trading data"""
        try:
            # For demo purposes, simulated congressional trading data
            # In production, this would scrape from House/Senate disclosure sites
            trades_data = {
                'recent_trades': [
                    {
                        'member': 'Representative A',
                        'trade_type': 'PURCHASE',
                        'asset': 'SPY',
                        'amount_range': '$50,001-$100,000',
                        'date': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'),
                        'disclosure_date': datetime.now().strftime('%Y-%m-%d'),
                        'market_impact': 'BULLISH'
                    },
                    {
                        'member': 'Senator B',
                        'trade_type': 'SALE',
                        'asset': 'QQQ',
                        'amount_range': '$15,001-$50,000',
                        'date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                        'disclosure_date': datetime.now().strftime('%Y-%m-%d'),
                        'market_impact': 'BEARISH'
                    }
                ],
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info(f"Fetched {len(trades_data['recent_trades'])} congressional trades")
            return trades_data
            
        except Exception as e:
            logger.error(f"Error fetching congressional trades: {e}")
            return {}
    
    def analyze_insider_signals(self, trades_data):
        """Analyze congressional trades for market signals"""
        try:
            if not trades_data or not trades_data.get('recent_trades'):
                return {}
            
            trades = trades_data['recent_trades']
            
            # Aggregate signals by market impact
            bullish_signals = len([t for t in trades if t['market_impact'] == 'BULLISH'])
            bearish_signals = len([t for t in trades if t['market_impact'] == 'BEARISH'])
            
            # Calculate net signal strength
            net_signal = bullish_signals - bearish_signals
            total_trades = len(trades)
            
            if net_signal > 0:
                bias = 'BULLISH'
                confidence = min(0.8, abs(net_signal) / max(total_trades, 1))
            elif net_signal < 0:
                bias = 'BEARISH'
                confidence = min(0.8, abs(net_signal) / max(total_trades, 1))
            else:
                bias = 'NEUTRAL'
                confidence = 0.5
            
            # Recent high-value trades get higher weight
            recent_high_value = [t for t in trades if '$50,001' in t['amount_range'] or '$100,000' in t['amount_range']]
            if recent_high_value:
                confidence *= 1.3
            
            analysis = {
                'congressional_bias': bias,
                'signal_strength': confidence,
                'total_trades': total_trades,
                'bullish_signals': bullish_signals,
                'bearish_signals': bearish_signals,
                'high_value_trades': len(recent_high_value),
                'trade_recommendation': {
                    'direction': bias,
                    'confidence': min(confidence, 0.9),
                    'timeframe': 'SHORT_TERM',  # Congressional effects are usually quick
                    'position_size_multiplier': 1.0 + (confidence * 0.3)
                },
                'key_trades': trades[:3]  # Top 3 most recent
            }
            
            logger.info(f"Congressional Analysis: {bias} bias with {confidence:.2f} confidence")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing congressional trades: {e}")
            return {}
    
    def save_analysis(self, analysis):
        """Save congressional analysis to file"""
        try:
            output_file = os.path.join(self.data_dir, 'latest_congressional_analysis.json')
            
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            logger.info(f"Saved congressional analysis to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving congressional analysis: {e}")

def main():
    """Main congressional trades monitoring workflow"""
    logger.info("Starting Congressional Trades monitoring...")
    
    monitor = CongressionalTradesMonitor()
    
    # Fetch and analyze congressional trades
    trades_data = monitor.fetch_congressional_trades()
    if trades_data:
        analysis = monitor.analyze_insider_signals(trades_data)
        if analysis:
            monitor.save_analysis(analysis)
            logger.info("Congressional trades analysis completed successfully")
        else:
            logger.error("Failed to analyze congressional trades")
    else:
        logger.error("Failed to fetch congressional trades data")

if __name__ == "__main__":
    main()
