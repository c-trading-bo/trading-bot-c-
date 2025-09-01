#!/usr/bin/env python3
"""
Sector Rotation Analysis for Intelligence Pipeline
Analyzes sector ETF performance and risk-on/off detection
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SectorRotationAnalyzer:
    def __init__(self):
        self.data_dir = "Intelligence/data/sectors"
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.sector_etfs = {
            'Technology': 'XLK',
            'Financials': 'XLF',
            'Energy': 'XLE',
            'Healthcare': 'XLV',
            'Consumer Discretionary': 'XLY',
            'Industrials': 'XLI',
            'Consumer Staples': 'XLP',
            'Utilities': 'XLU',
            'Materials': 'XLB',
            'Real Estate': 'XLRE',
            'Communication Services': 'XLC'
        }
        
    def analyze_sector_performance(self):
        """Analyze sector rotation patterns"""
        try:
            sector_data = {}
            
            for sector_name, etf_symbol in self.sector_etfs.items():
                ticker = yf.Ticker(etf_symbol)
                hist = ticker.history(period="30d")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    price_30d_ago = hist['Close'].iloc[0]
                    performance = ((current_price - price_30d_ago) / price_30d_ago) * 100
                    
                    sector_data[sector_name] = {
                        'symbol': etf_symbol,
                        'performance_30d': round(performance, 2),
                        'current_price': round(current_price, 2),
                        'relative_strength': self.calculate_relative_strength(hist)
                    }
            
            # Determine risk-on/off sentiment
            risk_on_sectors = ['Technology', 'Consumer Discretionary', 'Financials']
            risk_off_sectors = ['Utilities', 'Consumer Staples', 'Healthcare']
            
            risk_on_performance = np.mean([sector_data[s]['performance_30d'] 
                                         for s in risk_on_sectors if s in sector_data])
            risk_off_performance = np.mean([sector_data[s]['performance_30d'] 
                                          for s in risk_off_sectors if s in sector_data])
            
            rotation_analysis = {
                'sector_performance': sector_data,
                'risk_on_performance': round(risk_on_performance, 2),
                'risk_off_performance': round(risk_off_performance, 2),
                'risk_sentiment': 'risk_on' if risk_on_performance > risk_off_performance else 'risk_off',
                'rotation_strength': abs(risk_on_performance - risk_off_performance),
                'leading_sectors': self.identify_leading_sectors(sector_data),
                'lagging_sectors': self.identify_lagging_sectors(sector_data)
            }
            
            return rotation_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing sector rotation: {e}")
            return {}
    
    def calculate_relative_strength(self, hist):
        """Calculate relative strength vs SPY"""
        try:
            spy = yf.Ticker('SPY').history(period="30d")
            if not spy.empty and len(spy) == len(hist):
                sector_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1)
                spy_return = (spy['Close'].iloc[-1] / spy['Close'].iloc[0] - 1)
                return round((sector_return - spy_return) * 100, 2)
            return 0
        except:
            return 0
    
    def identify_leading_sectors(self, sector_data, top_n=3):
        """Identify leading sectors"""
        sorted_sectors = sorted(sector_data.items(), 
                              key=lambda x: x[1]['performance_30d'], reverse=True)
        return [sector for sector, _ in sorted_sectors[:top_n]]
    
    def identify_lagging_sectors(self, sector_data, bottom_n=3):
        """Identify lagging sectors"""
        sorted_sectors = sorted(sector_data.items(), 
                              key=lambda x: x[1]['performance_30d'])
        return [sector for sector, _ in sorted_sectors[:bottom_n]]

def main():
    """Main execution function"""
    logger.info("Starting sector rotation analysis...")
    
    analyzer = SectorRotationAnalyzer()
    analysis = analyzer.analyze_sector_performance()
    
    # Save results
    timestamp_str = datetime.now().strftime("%Y-%m-%d")
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'sector_rotation_analysis': analysis,
        'generated_by': 'sector_rotation_analyzer'
    }
    
    filename = f"{timestamp_str}.json"
    filepath = os.path.join(analyzer.data_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info("Sector rotation analysis completed")

if __name__ == "__main__":
    main()