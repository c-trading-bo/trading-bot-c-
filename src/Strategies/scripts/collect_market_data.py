#!/usr/bin/env python3
"""
Market Data Collector for Intelligence Pipeline
Collects SPX, VIX, and other market indices data after market close
"""

import yfinance as yf
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketDataCollector:
    def __init__(self):
        self.data_dir = "Intelligence/data/raw/indices"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Market symbols to track
        self.symbols = {
            'SPX': '^GSPC',  # S&P 500
            'VIX': '^VIX',   # VIX Volatility Index
            'NDX': '^NDX',   # NASDAQ 100
            'DJI': '^DJI',   # Dow Jones
            'RUT': '^RUT',   # Russell 2000
            'TNX': '^TNX',   # 10-Year Treasury
            'DXY': 'DX-Y.NYB'  # Dollar Index
        }
    
    def collect_daily_data(self, days_back=5):
        """Collect daily market data for the specified symbols"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            market_data = {}
            
            for symbol_name, symbol_code in self.symbols.items():
                try:
                    ticker = yf.Ticker(symbol_code)
                    hist = ticker.history(start=start_date, end=end_date)
                    
                    if hist.empty:
                        logger.warning(f"No data retrieved for {symbol_name}")
                        continue
                    
                    # Get the latest day's data
                    latest = hist.iloc[-1]
                    prev_close = hist.iloc[-2]['Close'] if len(hist) > 1 else latest['Close']
                    
                    market_data[symbol_name] = {
                        'symbol': symbol_code,
                        'date': latest.name.strftime('%Y-%m-%d'),
                        'open': float(latest['Open']),
                        'high': float(latest['High']),
                        'low': float(latest['Low']),
                        'close': float(latest['Close']),
                        'volume': int(latest['Volume']) if 'Volume' in latest else 0,
                        'prev_close': float(prev_close),
                        'change': float(latest['Close'] - prev_close),
                        'change_pct': float((latest['Close'] - prev_close) / prev_close * 100)
                    }
                    
                    # Calculate additional metrics
                    if len(hist) >= 5:
                        # 5-day metrics
                        hist_5d = hist.tail(5)
                        market_data[symbol_name]['volatility_5d'] = float(hist_5d['Close'].pct_change().std() * 100)
                        market_data[symbol_name]['trend_5d'] = float((hist_5d['Close'].iloc[-1] - hist_5d['Close'].iloc[0]) / hist_5d['Close'].iloc[0] * 100)
                    
                    if len(hist) >= 20:
                        # 20-day metrics
                        hist_20d = hist.tail(20)
                        market_data[symbol_name]['sma_20'] = float(hist_20d['Close'].mean())
                        market_data[symbol_name]['volatility_20d'] = float(hist_20d['Close'].pct_change().std() * 100)
                    
                    logger.info(f"Collected data for {symbol_name}: {latest['Close']:.2f} ({market_data[symbol_name]['change_pct']:+.2f}%)")
                    
                except Exception as e:
                    logger.error(f"Failed to collect data for {symbol_name}: {e}")
                    continue
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error collecting market data: {e}")
            return {}
    
    def analyze_market_regime(self, market_data):
        """Analyze current market regime based on collected data"""
        if not market_data or 'SPX' not in market_data or 'VIX' not in market_data:
            return {
                'regime': 'Unknown',
                'confidence': 0,
                'reasoning': 'Insufficient data'
            }
        
        spx_data = market_data['SPX']
        vix_data = market_data['VIX']
        
        # VIX-based regime classification
        vix_close = vix_data['close']
        spx_volatility = spx_data.get('volatility_20d', 0)
        
        if vix_close > 30:
            regime = 'Volatile'
            confidence = min(90, (vix_close - 20) * 2)
        elif vix_close < 15:
            regime = 'Trending'
            confidence = min(90, (25 - vix_close) * 3)
        else:
            # Check for ranging based on recent price action
            if spx_volatility < 1.5:
                regime = 'Ranging'
                confidence = 70
            else:
                regime = 'Trending'
                confidence = 60
        
        # Additional factors
        spx_change = spx_data['change_pct']
        trend_strength = abs(spx_change)
        
        reasoning_parts = [
            f"VIX: {vix_close:.1f}",
            f"SPX change: {spx_change:+.2f}%"
        ]
        
        if 'volatility_20d' in spx_data:
            reasoning_parts.append(f"20d volatility: {spx_data['volatility_20d']:.2f}%")
        
        analysis = {
            'regime': regime,
            'confidence': confidence,
            'reasoning': ', '.join(reasoning_parts),
            'vix_level': vix_close,
            'spx_change': spx_change,
            'trend_strength': trend_strength
        }
        
        return analysis
    
    def check_economic_calendar(self):
        """Simple check for major economic events (placeholder for more sophisticated calendar)"""
        # This is a simplified version - in practice, you'd integrate with an economic calendar API
        today = datetime.now()
        day_of_month = today.day
        
        # Simple heuristics for common event days
        is_cpi_week = 10 <= day_of_month <= 16  # CPI usually mid-month
        is_fomc_potential = day_of_month in [19, 20, 21] or day_of_month in [1, 2, 3]  # Common FOMC dates
        
        return {
            'is_cpi_day': is_cpi_week and today.weekday() < 5,  # Weekday in CPI week
            'is_fomc_day': is_fomc_potential and today.weekday() < 5,  # Weekday in FOMC potential days
            'calendar_checked': today.isoformat()
        }
    
    def save_market_data(self, market_data, regime_analysis, calendar_info):
        """Save collected market data and analysis"""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Combine all data
        full_data = {
            'timestamp': datetime.now().isoformat(),
            'market_data': market_data,
            'regime_analysis': regime_analysis,
            'economic_calendar': calendar_info
        }
        
        # Save timestamped file
        data_file = os.path.join(self.data_dir, f"market_data_{timestamp_str}.json")
        with open(data_file, 'w') as f:
            json.dump(full_data, f, indent=2)
        
        # Update latest file for bot consumption
        latest_file = os.path.join(self.data_dir, "latest_market_data.json")
        with open(latest_file, 'w') as f:
            json.dump(full_data, f, indent=2)
        
        logger.info(f"Saved market data: regime={regime_analysis['regime']}, confidence={regime_analysis['confidence']:.1f}%")
        
        return data_file

def main():
    """Main execution function"""
    logger.info("Starting market data collection...")
    
    collector = MarketDataCollector()
    
    # Collect market data
    market_data = collector.collect_daily_data()
    
    if not market_data:
        logger.error("Failed to collect market data")
        return
    
    # Analyze market regime
    regime_analysis = collector.analyze_market_regime(market_data)
    
    # Check economic calendar
    calendar_info = collector.check_economic_calendar()
    
    # Save all data
    collector.save_market_data(market_data, regime_analysis, calendar_info)
    
    logger.info(f"Market data collection completed: {len(market_data)} symbols processed")

if __name__ == "__main__":
    main()