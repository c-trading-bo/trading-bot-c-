#!/usr/bin/env python3
"""
Intermarket Correlations Analysis for Intelligence Pipeline
Analyzes correlations between ES/NQ and DXY, TNX, Oil, Gold, Bitcoin
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import logging
from scipy.stats import pearsonr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CorrelationAnalyzer:
    def __init__(self):
        self.data_dir = "Intelligence/data/correlations"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Define instruments
        self.equity_symbols = ['ES=F', 'NQ=F']
        self.intermarket_symbols = {
            'DXY': 'DX=F',      # Dollar Index
            'TNX': '^TNX',      # 10-Year Treasury
            'OIL': 'CL=F',      # Crude Oil
            'GOLD': 'GC=F',     # Gold
            'BTC': 'BTC-USD'    # Bitcoin
        }
        
    def fetch_data(self, period="30d"):
        """Fetch data for all instruments"""
        try:
            data = {}
            
            # Fetch equity data
            for symbol in self.equity_symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period, interval="1h")
                if not hist.empty:
                    data[symbol] = hist['Close'].pct_change().dropna()
            
            # Fetch intermarket data
            for name, symbol in self.intermarket_symbols.items():
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period, interval="1h")
                if not hist.empty:
                    data[name] = hist['Close'].pct_change().dropna()
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return {}
    
    def calculate_correlations(self, data, lookback_periods=[24, 168, 720]):  # 1d, 1w, 1m in hours
        """Calculate rolling correlations"""
        try:
            correlation_results = {}
            
            for equity in self.equity_symbols:
                if equity not in data:
                    continue
                    
                correlation_results[equity] = {}
                
                for market_name in self.intermarket_symbols.keys():
                    if market_name not in data:
                        continue
                    
                    correlation_results[equity][market_name] = {}
                    
                    # Align data
                    df = pd.DataFrame({
                        'equity': data[equity],
                        'market': data[market_name]
                    }).dropna()
                    
                    if len(df) < max(lookback_periods):
                        continue
                    
                    # Calculate correlations for different periods
                    for period in lookback_periods:
                        if len(df) >= period:
                            recent_data = df.tail(period)
                            corr, p_value = pearsonr(recent_data['equity'], recent_data['market'])
                            
                            period_name = f"{period}h"
                            correlation_results[equity][market_name][period_name] = {
                                'correlation': round(corr, 3),
                                'p_value': round(p_value, 3),
                                'significance': 'significant' if p_value < 0.05 else 'not_significant',
                                'strength': self.classify_correlation_strength(abs(corr))
                            }
            
            return correlation_results
            
        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
            return {}
    
    def classify_correlation_strength(self, corr):
        """Classify correlation strength"""
        if corr >= 0.7:
            return 'strong'
        elif corr >= 0.3:
            return 'moderate'
        elif corr >= 0.1:
            return 'weak'
        else:
            return 'negligible'
    
    def detect_regime_changes(self, correlation_results):
        """Detect correlation regime changes"""
        try:
            regime_signals = []
            
            for equity, markets in correlation_results.items():
                for market, periods in markets.items():
                    if '24h' in periods and '168h' in periods:
                        short_corr = periods['24h']['correlation']
                        long_corr = periods['168h']['correlation']
                        
                        # Detect significant changes
                        change = abs(short_corr - long_corr)
                        if change > 0.3:  # Significant change threshold
                            direction = 'increasing' if short_corr > long_corr else 'decreasing'
                            regime_signals.append({
                                'equity': equity,
                                'market': market,
                                'type': 'correlation_regime_change',
                                'direction': direction,
                                'short_term_corr': short_corr,
                                'long_term_corr': long_corr,
                                'change_magnitude': round(change, 3)
                            })
            
            return regime_signals
            
        except Exception as e:
            logger.error(f"Error detecting regime changes: {e}")
            return []
    
    def generate_signals(self, correlation_results, regime_signals):
        """Generate trading signals based on correlations"""
        try:
            signals = []
            
            # Check for correlation extremes
            for equity, markets in correlation_results.items():
                for market, periods in markets.items():
                    if '24h' in periods:
                        corr = periods['24h']['correlation']
                        
                        # Strong positive correlation signals
                        if corr > 0.7 and market in ['DXY', 'TNX']:
                            signals.append({
                                'type': 'correlation_extreme',
                                'equity': equity,
                                'market': market,
                                'correlation': corr,
                                'signal': f'Strong positive correlation with {market} - consider {equity} weakness if {market} strengthens',
                                'confidence': 'high'
                            })
                        
                        # Strong negative correlation signals
                        elif corr < -0.7 and market in ['GOLD', 'BTC']:
                            signals.append({
                                'type': 'correlation_extreme',
                                'equity': equity,
                                'market': market,
                                'correlation': corr,
                                'signal': f'Strong negative correlation with {market} - consider {equity} strength if {market} weakens',
                                'confidence': 'high'
                            })
            
            # Add regime change signals
            for regime_signal in regime_signals:
                signals.append({
                    'type': 'regime_change',
                    'signal': f"Correlation regime change detected: {regime_signal['equity']} vs {regime_signal['market']}",
                    'details': regime_signal,
                    'confidence': 'medium'
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    def save_analysis(self, correlation_results, regime_signals, trading_signals):
        """Save correlation analysis"""
        try:
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M")
            
            output_data = {
                'timestamp': datetime.now().isoformat(),
                'correlation_matrix': correlation_results,
                'regime_signals': regime_signals,
                'trading_signals': trading_signals,
                'summary': {
                    'correlations_calculated': sum(len(markets) for markets in correlation_results.values()),
                    'regime_changes_detected': len(regime_signals),
                    'trading_signals_generated': len(trading_signals)
                },
                'generated_by': 'correlation_analyzer'
            }
            
            # Save timestamped file
            filename = f"{timestamp_str}.json"
            filepath = os.path.join(self.data_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Correlation analysis saved: {output_data['summary']['correlations_calculated']} correlations, {len(trading_signals)} signals")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
            return None

def main():
    """Main execution function"""
    logger.info("Starting intermarket correlation analysis...")
    
    analyzer = CorrelationAnalyzer()
    
    # Fetch data
    data = analyzer.fetch_data()
    
    if not data:
        logger.error("No data available for analysis")
        return
    
    # Calculate correlations
    correlation_results = analyzer.calculate_correlations(data)
    
    # Detect regime changes
    regime_signals = analyzer.detect_regime_changes(correlation_results)
    
    # Generate trading signals
    trading_signals = analyzer.generate_signals(correlation_results, regime_signals)
    
    # Save analysis
    filepath = analyzer.save_analysis(correlation_results, regime_signals, trading_signals)
    
    if filepath:
        logger.info("Intermarket correlation analysis completed successfully")
    else:
        logger.error("Intermarket correlation analysis failed")

if __name__ == "__main__":
    main()