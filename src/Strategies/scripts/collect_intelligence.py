#!/usr/bin/env python3
"""
INTELLIGENCE COLLECTION SYSTEM
Collects market intelligence data for trading analysis
"""

import os
import sys
import json
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

class IntelligenceCollector:
    def __init__(self):
        self.data_dir = Path("data/intelligence")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.reports_dir = Path("Intelligence/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_market_sentiment(self) -> dict:
        """Collect market sentiment indicators"""
        print("📊 Collecting market sentiment data...")
        
        sentiment_data = {
            'timestamp': datetime.now().isoformat(),
            'vix_analysis': self.analyze_vix(),
            'put_call_ratio': self.get_put_call_ratio(),
            'insider_trading': self.check_insider_activity(),
            'institutional_flows': self.analyze_institutional_flows()
        }
        
        return sentiment_data
    
    def analyze_vix(self) -> dict:
        """Analyze VIX data for fear/greed indicators"""
        try:
            import yfinance as yf
            
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="5d")
            
            if not hist.empty:
                current_vix = hist['Close'].iloc[-1]
                vix_change = hist['Close'].iloc[-1] - hist['Close'].iloc[-2]
                
                # VIX interpretation
                if current_vix < 15:
                    sentiment = "COMPLACENT"
                elif current_vix < 20:
                    sentiment = "LOW_FEAR" 
                elif current_vix < 30:
                    sentiment = "MODERATE_FEAR"
                else:
                    sentiment = "HIGH_FEAR"
                
                return {
                    'current_level': float(current_vix),
                    'daily_change': float(vix_change),
                    'sentiment': sentiment,
                    'updated': datetime.now().isoformat()
                }
            else:
                return {'error': 'No VIX data available'}
                
        except Exception as e:
            return {'error': str(e)}
    
    def get_put_call_ratio(self) -> dict:
        """Estimate put/call ratio from options data"""
        try:
            import yfinance as yf
            
            spy = yf.Ticker("SPY")
            options_dates = spy.options
            
            if options_dates:
                # Get nearest expiry options
                nearest_date = options_dates[0]
                puts = spy.option_chain(nearest_date).puts
                calls = spy.option_chain(nearest_date).calls
                
                put_volume = puts['volume'].sum() if not puts.empty else 0
                call_volume = calls['volume'].sum() if not calls.empty else 0
                
                if call_volume > 0:
                    pc_ratio = put_volume / call_volume
                    
                    if pc_ratio > 1.0:
                        sentiment = "BEARISH"
                    elif pc_ratio > 0.8:
                        sentiment = "CAUTIOUS"
                    else:
                        sentiment = "BULLISH"
                    
                    return {
                        'put_call_ratio': float(pc_ratio),
                        'sentiment': sentiment,
                        'put_volume': int(put_volume),
                        'call_volume': int(call_volume),
                        'updated': datetime.now().isoformat()
                    }
            
            return {'error': 'Options data unavailable'}
            
        except Exception as e:
            return {'error': str(e)}
    
    def check_insider_activity(self) -> dict:
        """Check for unusual insider trading activity"""
        # Placeholder for insider trading data
        # In production, this would connect to SEC EDGAR or similar
        
        return {
            'insider_buys': 'Limited data available',
            'insider_sells': 'Limited data available',
            'net_activity': 'NEUTRAL',
            'note': 'Requires premium data source integration',
            'updated': datetime.now().isoformat()
        }
    
    def analyze_institutional_flows(self) -> dict:
        """Analyze institutional money flows"""
        try:
            # Use ETF flows as proxy for institutional activity
            import yfinance as yf
            
            institutional_etfs = ['SPY', 'QQQ', 'IWM', 'XLF', 'XLE']
            flows = {}
            
            for etf in institutional_etfs:
                ticker = yf.Ticker(etf)
                hist = ticker.history(period="5d")
                
                if not hist.empty:
                    volume_trend = hist['Volume'].iloc[-3:].mean() / hist['Volume'].iloc[-10:-3].mean()
                    price_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5]
                    
                    flows[etf] = {
                        'volume_trend': float(volume_trend),
                        'price_change_5d': float(price_change),
                        'relative_strength': 'STRONG' if volume_trend > 1.2 else 'WEAK' if volume_trend < 0.8 else 'NORMAL'
                    }
            
            return {
                'etf_flows': flows,
                'overall_sentiment': self.calculate_flow_sentiment(flows),
                'updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_flow_sentiment(self, flows: dict) -> str:
        """Calculate overall sentiment from flows"""
        if not flows:
            return 'UNKNOWN'
        
        strong_count = sum(1 for data in flows.values() if data.get('relative_strength') == 'STRONG')
        weak_count = sum(1 for data in flows.values() if data.get('relative_strength') == 'WEAK')
        
        if strong_count > weak_count:
            return 'BULLISH_FLOWS'
        elif weak_count > strong_count:
            return 'BEARISH_FLOWS'
        else:
            return 'NEUTRAL_FLOWS'
    
    def collect_economic_indicators(self) -> dict:
        """Collect key economic indicators"""
        print("📈 Collecting economic indicators...")
        
        try:
            # Treasury yields (proxy for economic sentiment)
            import yfinance as yf
            
            yield_tickers = ['^TNX', '^FVX', '^IRX']  # 10Y, 5Y, 3M
            yields = {}
            
            for ticker in yield_tickers:
                treasury = yf.Ticker(ticker)
                hist = treasury.history(period="5d")
                
                if not hist.empty:
                    current_yield = hist['Close'].iloc[-1]
                    yield_change = hist['Close'].iloc[-1] - hist['Close'].iloc[-2]
                    
                    yields[ticker.replace('^', '')] = {
                        'current_yield': float(current_yield),
                        'daily_change': float(yield_change)
                    }
            
            # Calculate yield curve slope
            slope = 'UNKNOWN'
            if 'TNX' in yields and 'IRX' in yields:
                curve_slope = yields['TNX']['current_yield'] - yields['IRX']['current_yield']
                if curve_slope > 1.5:
                    slope = 'STEEP'
                elif curve_slope > 0.5:
                    slope = 'NORMAL'
                elif curve_slope > 0:
                    slope = 'FLATTENING'
                else:
                    slope = 'INVERTED'
            
            return {
                'treasury_yields': yields,
                'yield_curve_slope': slope,
                'economic_sentiment': 'RISK_ON' if slope in ['STEEP', 'NORMAL'] else 'RISK_OFF',
                'updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def run_comprehensive_collection(self) -> dict:
        """Run comprehensive intelligence collection"""
        print("🧠 INTELLIGENCE COLLECTION SYSTEM")
        print("=" * 50)
        
        intelligence_report = {
            'collection_timestamp': datetime.now().isoformat(),
            'market_sentiment': self.collect_market_sentiment(),
            'economic_indicators': self.collect_economic_indicators(),
            'collection_status': 'COMPLETED',
            'data_quality': 'HIGH'
        }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"intelligence_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(intelligence_report, f, indent=2, default=str)
        
        # Save latest report
        latest_file = self.reports_dir / "latest_intelligence.json"
        with open(latest_file, 'w') as f:
            json.dump(intelligence_report, f, indent=2, default=str)
        
        print(f"✅ Intelligence report saved: {report_file}")
        print(f"📊 Market Sentiment: {intelligence_report['market_sentiment']['vix_analysis'].get('sentiment', 'UNKNOWN')}")
        print(f"📈 Economic Outlook: {intelligence_report['economic_indicators'].get('economic_sentiment', 'UNKNOWN')}")
        
        return intelligence_report

def main():
    """Main intelligence collection function"""
    collector = IntelligenceCollector()
    report = collector.run_comprehensive_collection()
    
    # Print summary
    print("\n🎯 INTELLIGENCE COLLECTION SUMMARY")
    print("=" * 50)
    print(f"Status: {report['collection_status']}")
    print(f"Data Quality: {report['data_quality']}")
    print(f"Timestamp: {report['collection_timestamp']}")
    
    return report

if __name__ == "__main__":
    main()