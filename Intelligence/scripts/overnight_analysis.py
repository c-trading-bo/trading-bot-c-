#!/usr/bin/env python3
"""
Overnight Session Analysis for Intelligence Pipeline
Analyzes Asian/European session performance and global market impact
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

class OvernightAnalyzer:
    def __init__(self):
        self.data_dir = "Intelligence/data/overnight"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Define global markets
        self.markets = {
            'US_FUTURES': {
                'ES': 'ES=F',
                'NQ': 'NQ=F',
                'YM': 'YM=F'
            },
            'ASIAN': {
                'NIKKEI': '^N225',
                'HANG_SENG': '^HSI',
                'SHANGHAI': '000001.SS'
            },
            'EUROPEAN': {
                'DAX': '^GDAXI',
                'FTSE': '^FTSE',
                'CAC': '^FCHI'
            },
            'COMMODITIES': {
                'OIL': 'CL=F',
                'GOLD': 'GC=F'
            }
        }
        
    def fetch_overnight_data(self):
        """Fetch overnight session data"""
        try:
            data = {}
            
            for region, symbols in self.markets.items():
                data[region] = {}
                
                for name, symbol in symbols.items():
                    ticker = yf.Ticker(symbol)
                    
                    # Get 5 days of hourly data
                    hist = ticker.history(period="5d", interval="1h")
                    
                    if not hist.empty:
                        data[region][name] = {
                            'data': hist,
                            'current_price': hist['Close'].iloc[-1],
                            'previous_close': hist['Close'].iloc[-25] if len(hist) > 25 else hist['Close'].iloc[0]  # ~1 day ago
                        }
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching overnight data: {e}")
            return {}
    
    def analyze_overnight_performance(self, data):
        """Analyze overnight session performance"""
        try:
            analysis = {}
            
            # Analyze each region
            for region, markets in data.items():
                if not markets:
                    continue
                    
                region_analysis = {
                    'performance': {},
                    'sentiment': 'neutral',
                    'strength': 0
                }
                
                performances = []
                
                for market_name, market_data in markets.items():
                    current = market_data['current_price']
                    previous = market_data['previous_close']
                    
                    # Calculate performance
                    pct_change = ((current - previous) / previous) * 100
                    performances.append(pct_change)
                    
                    region_analysis['performance'][market_name] = {
                        'current_price': round(current, 2),
                        'previous_close': round(previous, 2),
                        'change_percent': round(pct_change, 2),
                        'direction': 'up' if pct_change > 0 else 'down'
                    }
                
                # Calculate region sentiment
                if performances:
                    avg_performance = np.mean(performances)
                    region_analysis['average_performance'] = round(avg_performance, 2)
                    region_analysis['strength'] = round(abs(avg_performance), 2)
                    
                    if avg_performance > 0.5:
                        region_analysis['sentiment'] = 'bullish'
                    elif avg_performance < -0.5:
                        region_analysis['sentiment'] = 'bearish'
                    else:
                        region_analysis['sentiment'] = 'neutral'
                
                analysis[region] = region_analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing overnight performance: {e}")
            return {}
    
    def calculate_gap_probabilities(self, data):
        """Calculate gap probabilities for US markets"""
        try:
            gap_analysis = {}
            
            if 'US_FUTURES' not in data:
                return gap_analysis
            
            for market_name, market_data in data['US_FUTURES'].items():
                hist = market_data['data']
                
                if len(hist) < 48:  # Need at least 2 days
                    continue
                
                # Identify gaps (simplified - using hourly data)
                gaps = []
                daily_closes = hist.resample('D')['Close'].last().dropna()
                daily_opens = hist.resample('D')['Open'].first().dropna()
                
                for i in range(1, len(daily_closes)):
                    if i < len(daily_opens):
                        prev_close = daily_closes.iloc[i-1]
                        next_open = daily_opens.iloc[i]
                        
                        gap_size = ((next_open - prev_close) / prev_close) * 100
                        if abs(gap_size) > 0.1:  # Only significant gaps
                            gaps.append(gap_size)
                
                if gaps:
                    gap_analysis[market_name] = {
                        'average_gap': round(np.mean(gaps), 3),
                        'gap_frequency': len(gaps) / max(len(daily_closes) - 1, 1),
                        'upward_gaps': len([g for g in gaps if g > 0]),
                        'downward_gaps': len([g for g in gaps if g < 0]),
                        'largest_gap': round(max(gaps, key=abs), 3) if gaps else 0
                    }
            
            return gap_analysis
            
        except Exception as e:
            logger.error(f"Error calculating gap probabilities: {e}")
            return {}
    
    def generate_signals(self, overnight_analysis, gap_analysis):
        """Generate signals based on overnight analysis"""
        try:
            signals = []
            
            # Global sentiment signals
            regions_bullish = sum(1 for region in overnight_analysis.values() if region.get('sentiment') == 'bullish')
            regions_bearish = sum(1 for region in overnight_analysis.values() if region.get('sentiment') == 'bearish')
            
            if regions_bullish >= 2:
                signals.append({
                    'type': 'global_sentiment',
                    'signal': 'Bullish global sentiment - multiple regions positive',
                    'confidence': 'high' if regions_bullish >= 3 else 'medium',
                    'regions_bullish': regions_bullish,
                    'regions_bearish': regions_bearish
                })
            elif regions_bearish >= 2:
                signals.append({
                    'type': 'global_sentiment',
                    'signal': 'Bearish global sentiment - multiple regions negative',
                    'confidence': 'high' if regions_bearish >= 3 else 'medium',
                    'regions_bullish': regions_bullish,
                    'regions_bearish': regions_bearish
                })
            
            # Regional strength signals
            for region, analysis in overnight_analysis.items():
                if analysis.get('strength', 0) > 1.0:  # Strong move
                    signals.append({
                        'type': 'regional_strength',
                        'region': region,
                        'signal': f'Strong {analysis.get("sentiment", "neutral")} move in {region}',
                        'strength': analysis.get('strength', 0),
                        'average_performance': analysis.get('average_performance', 0)
                    })
            
            # Gap signals
            for market, gap_data in gap_analysis.items():
                if gap_data.get('gap_frequency', 0) > 0.3:  # High gap frequency
                    signals.append({
                        'type': 'gap_probability',
                        'market': market,
                        'signal': f'High gap probability for {market}',
                        'gap_frequency': gap_data.get('gap_frequency', 0),
                        'average_gap': gap_data.get('average_gap', 0)
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    def save_analysis(self, overnight_analysis, gap_analysis, signals):
        """Save overnight analysis"""
        try:
            timestamp_str = datetime.now().strftime("%Y-%m-%d")
            
            output_data = {
                'timestamp': datetime.now().isoformat(),
                'overnight_performance': overnight_analysis,
                'gap_analysis': gap_analysis,
                'signals': signals,
                'summary': {
                    'regions_analyzed': len(overnight_analysis),
                    'markets_analyzed': sum(len(region.get('performance', {})) for region in overnight_analysis.values()),
                    'signals_generated': len(signals)
                },
                'generated_by': 'overnight_analyzer'
            }
            
            # Save daily file
            filename = f"{timestamp_str}.json"
            filepath = os.path.join(self.data_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Overnight analysis saved: {output_data['summary']['regions_analyzed']} regions, {len(signals)} signals")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
            return None

def main():
    """Main execution function"""
    logger.info("Starting overnight session analysis...")
    
    analyzer = OvernightAnalyzer()
    
    # Fetch overnight data
    data = analyzer.fetch_overnight_data()
    
    if not data:
        logger.error("No overnight data available")
        return
    
    # Analyze overnight performance
    overnight_analysis = analyzer.analyze_overnight_performance(data)
    
    # Calculate gap probabilities
    gap_analysis = analyzer.calculate_gap_probabilities(data)
    
    # Generate signals
    signals = analyzer.generate_signals(overnight_analysis, gap_analysis)
    
    # Save analysis
    filepath = analyzer.save_analysis(overnight_analysis, gap_analysis, signals)
    
    if filepath:
        logger.info("Overnight session analysis completed successfully")
    else:
        logger.error("Overnight session analysis failed")

if __name__ == "__main__":
    main()