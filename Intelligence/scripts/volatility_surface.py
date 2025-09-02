#!/usr/bin/env python3
"""
Volatility Surface Analysis for Intelligence Pipeline
Analyzes VIX term structure and volatility regime
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

class VolatilitySurfaceAnalyzer:
    def __init__(self):
        self.data_dir = "Intelligence/data/volatility"
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.vix_symbols = {
            'VIX9D': '^VIX9D',   # 9-day VIX
            'VIX': '^VIX',       # 30-day VIX
            'VIX3M': '^VIX3M',   # 3-month VIX
            'VIX6M': '^VIX6M'    # 6-month VIX
        }
        
    def fetch_vix_data(self):
        """Fetch VIX term structure data"""
        try:
            vix_data = {}
            
            for name, symbol in self.vix_symbols.items():
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="30d")
                
                if not hist.empty:
                    vix_data[name] = {
                        'current': hist['Close'].iloc[-1],
                        'history': hist['Close'],
                        'percentile_30d': self.calculate_percentile(hist['Close'], hist['Close'].iloc[-1])
                    }
            
            return vix_data
            
        except Exception as e:
            logger.error(f"Error fetching VIX data: {e}")
            return {}
    
    def calculate_percentile(self, series, current_value):
        """Calculate percentile of current value in series"""
        return (series < current_value).mean() * 100
    
    def analyze_term_structure(self, vix_data):
        """Analyze VIX term structure"""
        try:
            analysis = {}
            
            if len(vix_data) < 2:
                return analysis
            
            # Get current levels
            levels = {}
            for name, data in vix_data.items():
                levels[name] = data['current']
            
            # Analyze structure
            analysis['current_levels'] = levels
            analysis['structure_type'] = self.classify_structure(levels)
            analysis['term_structure_slope'] = self.calculate_slope(levels)
            analysis['backwardation_signals'] = self.detect_backwardation(levels)
            
            # Calculate regime
            if 'VIX' in levels:
                vix_level = levels['VIX']
                analysis['volatility_regime'] = self.classify_volatility_regime(vix_level)
                analysis['vix_percentile'] = vix_data['VIX']['percentile_30d']
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing term structure: {e}")
            return {}
    
    def classify_structure(self, levels):
        """Classify term structure type"""
        if 'VIX9D' in levels and 'VIX' in levels and 'VIX3M' in levels:
            if levels['VIX9D'] < levels['VIX'] < levels['VIX3M']:
                return 'normal_contango'
            elif levels['VIX9D'] > levels['VIX'] > levels['VIX3M']:
                return 'backwardation'
            elif levels['VIX9D'] > levels['VIX'] < levels['VIX3M']:
                return 'mixed'
            else:
                return 'steep_contango'
        return 'unknown'
    
    def calculate_slope(self, levels):
        """Calculate term structure slope"""
        if 'VIX9D' in levels and 'VIX3M' in levels:
            return round((levels['VIX3M'] - levels['VIX9D']) / levels['VIX9D'] * 100, 2)
        return 0
    
    def detect_backwardation(self, levels):
        """Detect backwardation signals"""
        signals = []
        
        if 'VIX9D' in levels and 'VIX' in levels:
            if levels['VIX9D'] > levels['VIX']:
                signals.append('Short-term backwardation detected')
        
        if 'VIX' in levels and 'VIX3M' in levels:
            if levels['VIX'] > levels['VIX3M']:
                signals.append('Medium-term backwardation detected')
        
        return signals
    
    def classify_volatility_regime(self, vix_level):
        """Classify current volatility regime"""
        if vix_level < 15:
            return 'low_volatility'
        elif vix_level < 20:
            return 'normal_volatility'
        elif vix_level < 30:
            return 'elevated_volatility'
        else:
            return 'high_volatility'
    
    def generate_signals(self, analysis):
        """Generate volatility-based signals"""
        try:
            signals = []
            
            # VIX level signals
            if 'current_levels' in analysis and 'VIX' in analysis['current_levels']:
                vix = analysis['current_levels']['VIX']
                percentile = analysis.get('vix_percentile', 50)
                
                if vix < 15 and percentile < 20:
                    signals.append({
                        'type': 'volatility_extreme',
                        'signal': 'VIX at extreme low - consider protective strategies',
                        'vix_level': vix,
                        'percentile': percentile,
                        'confidence': 'high'
                    })
                elif vix > 30 and percentile > 80:
                    signals.append({
                        'type': 'volatility_extreme',
                        'signal': 'VIX at extreme high - consider mean reversion',
                        'vix_level': vix,
                        'percentile': percentile,
                        'confidence': 'high'
                    })
            
            # Term structure signals
            if analysis.get('structure_type') == 'backwardation':
                signals.append({
                    'type': 'term_structure',
                    'signal': 'VIX backwardation - expect continued volatility',
                    'structure': analysis['structure_type'],
                    'confidence': 'medium'
                })
            
            # Regime signals
            regime = analysis.get('volatility_regime')
            if regime in ['low_volatility', 'high_volatility']:
                signals.append({
                    'type': 'volatility_regime',
                    'signal': f'Volatility regime: {regime.replace("_", " ")}',
                    'regime': regime,
                    'confidence': 'medium'
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    def save_analysis(self, vix_data, analysis, signals):
        """Save volatility analysis"""
        try:
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H")
            
            output_data = {
                'timestamp': datetime.now().isoformat(),
                'vix_data': {name: {'current': data['current'], 'percentile': data['percentile_30d']} 
                           for name, data in vix_data.items()},
                'term_structure_analysis': analysis,
                'signals': signals,
                'summary': {
                    'vix_instruments': len(vix_data),
                    'signals_generated': len(signals),
                    'volatility_regime': analysis.get('volatility_regime', 'unknown')
                },
                'generated_by': 'volatility_surface_analyzer'
            }
            
            # Save hourly file
            filename = f"{timestamp_str}.json"
            filepath = os.path.join(self.data_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Volatility analysis saved: regime={analysis.get('volatility_regime', 'unknown')}, signals={len(signals)}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
            return None

def main():
    """Main execution function"""
    logger.info("Starting volatility surface analysis...")
    
    analyzer = VolatilitySurfaceAnalyzer()
    
    # Fetch VIX data
    vix_data = analyzer.fetch_vix_data()
    
    if not vix_data:
        logger.error("No VIX data available")
        return
    
    # Analyze term structure
    analysis = analyzer.analyze_term_structure(vix_data)
    
    # Generate signals
    signals = analyzer.generate_signals(analysis)
    
    # Save analysis
    filepath = analyzer.save_analysis(vix_data, analysis, signals)
    
    if filepath:
        logger.info("Volatility surface analysis completed successfully")
    else:
        logger.error("Volatility surface analysis failed")

if __name__ == "__main__":
    main()