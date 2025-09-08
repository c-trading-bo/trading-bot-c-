#!/usr/bin/env python3
"""
ES/NQ Options Flow Integration (CRITICAL FOR FUTURES)
Real-time options flow analysis using SPY/QQQ as proxies for ES/NQ futures trading
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ES_NQ_OptionsFlowAnalyzer:
    """Real-time options flow for ES/NQ futures trading"""
    
    def __init__(self):
        self.symbols = {
            'ES': ['SPY', 'SPX', 'ES=F'],  # ES proxies
            'NQ': ['QQQ', 'NDX', 'NQ=F']   # NQ proxies
        }
        self.gamma_levels = {}
        self.dealer_positioning = {}
        
        # Create output directory
        self.output_dir = "Intelligence/data/options"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def analyze_realtime_flow(self):
        """Analyze options flow every 5 minutes during market hours"""
        
        flow_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'ES': {},
            'NQ': {}
        }
        
        try:
            # ES OPTIONS FLOW (SPY as proxy)
            logger.info("Analyzing ES options flow via SPY...")
            spy_chain = self.get_options_chain('SPY')
            if spy_chain:
                es_flow = self.analyze_chain(spy_chain, 'ES')
                flow_data['ES'] = {
                    'gamma_wall': es_flow['gamma_wall'],
                    'put_call_ratio': es_flow['pc_ratio'],
                    'dealer_positioning': es_flow['dealer_pos'],
                    'unusual_activity': es_flow['unusual'],
                    'key_strikes': es_flow['key_strikes'],
                    'sentiment': self.calculate_sentiment(es_flow)
                }
            
            # NQ OPTIONS FLOW (QQQ as proxy)
            logger.info("Analyzing NQ options flow via QQQ...")
            qqq_chain = self.get_options_chain('QQQ')
            if qqq_chain:
                nq_flow = self.analyze_chain(qqq_chain, 'NQ')
                flow_data['NQ'] = {
                    'gamma_wall': nq_flow['gamma_wall'],
                    'put_call_ratio': nq_flow['pc_ratio'],
                    'dealer_positioning': nq_flow['dealer_pos'],
                    'unusual_activity': nq_flow['unusual'],
                    'key_strikes': nq_flow['key_strikes'],
                    'sentiment': self.calculate_sentiment(nq_flow)
                }
            
            # GENERATE ES/NQ SIGNALS
            signals = self.generate_futures_signals(flow_data)
            flow_data['signals'] = signals
            
            return flow_data
            
        except Exception as e:
            logger.error(f"Error in realtime flow analysis: {e}")
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e),
                'ES': {'sentiment': 'NEUTRAL'},
                'NQ': {'sentiment': 'NEUTRAL'},
                'signals': {'ES': {'action': 'NEUTRAL', 'confidence': 0}, 'NQ': {'action': 'NEUTRAL', 'confidence': 0}}
            }
    
    def get_options_chain(self, symbol):
        """Get options chain data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            
            if not expirations:
                logger.warning(f"No options data available for {symbol}")
                return None
                
            # Get nearest expiration
            nearest_exp = expirations[0]
            
            # Get calls and puts
            chain = ticker.option_chain(nearest_exp)
            
            return {
                'calls': chain.calls,
                'puts': chain.puts,
                'expiration': nearest_exp
            }
            
        except Exception as e:
            logger.error(f"Error getting options chain for {symbol}: {e}")
            return None
    
    def analyze_chain(self, chain, instrument):
        """Analyze options chain for futures signals"""
        
        try:
            calls = chain['calls']
            puts = chain['puts']
            
            # Calculate gamma exposure
            gamma_exposure = self.calculate_gamma_exposure(calls, puts)
            
            # Find gamma walls (support/resistance)
            gamma_wall = self.find_gamma_walls(gamma_exposure)
            
            # Calculate put/call ratios
            pc_ratio = self.calculate_pc_ratio(calls, puts)
            
            # Detect unusual activity
            unusual = self.detect_unusual_activity(calls, puts)
            
            # Dealer positioning
            dealer_pos = self.calculate_dealer_positioning(calls, puts)
            
            # Key strikes with high OI
            key_strikes = self.find_key_strikes(calls, puts)
            
            return {
                'gamma_wall': gamma_wall,
                'pc_ratio': pc_ratio,
                'dealer_pos': dealer_pos,
                'unusual': unusual,
                'key_strikes': key_strikes,
                'gamma_exposure': gamma_exposure
            }
            
        except Exception as e:
            logger.error(f"Error analyzing chain for {instrument}: {e}")
            return {
                'gamma_wall': 0,
                'pc_ratio': 1.0,
                'dealer_pos': 0,
                'unusual': 0,
                'key_strikes': [],
                'gamma_exposure': {}
            }
    
    def calculate_gamma_exposure(self, calls, puts):
        """Calculate GEX for each strike"""
        
        gamma_profile = {}
        
        try:
            # Process calls
            for _, row in calls.iterrows():
                strike = row.get('strike', 0)
                gamma = row.get('gamma', 0)
                oi = row.get('openInterest', 0)
                
                if strike > 0:
                    call_gamma = gamma * oi * 100
                    gamma_profile[strike] = gamma_profile.get(strike, 0) + call_gamma
            
            # Process puts (negative gamma for dealers)
            for _, row in puts.iterrows():
                strike = row.get('strike', 0)
                gamma = row.get('gamma', 0)
                oi = row.get('openInterest', 0)
                
                if strike > 0:
                    put_gamma = gamma * oi * 100
                    gamma_profile[strike] = gamma_profile.get(strike, 0) - put_gamma
                    
        except Exception as e:
            logger.error(f"Error calculating gamma exposure: {e}")
        
        return gamma_profile
    
    def find_gamma_walls(self, gamma_exposure):
        """Find the largest gamma levels (support/resistance)"""
        
        if not gamma_exposure:
            return 0
            
        # Find strike with maximum absolute gamma
        max_gamma_strike = max(gamma_exposure.items(), key=lambda x: abs(x[1]))
        return max_gamma_strike[0] if max_gamma_strike else 0
    
    def calculate_pc_ratio(self, calls, puts):
        """Calculate put/call ratio by volume"""
        
        try:
            call_volume = calls['volume'].fillna(0).sum()
            put_volume = puts['volume'].fillna(0).sum()
            
            if call_volume > 0:
                return put_volume / call_volume
            else:
                return 1.0
                
        except Exception:
            return 1.0
    
    def detect_unusual_activity(self, calls, puts):
        """Detect unusual options activity"""
        
        try:
            # Calculate volume to OI ratios
            call_vol_oi = (calls['volume'].fillna(0) / calls['openInterest'].fillna(1)).fillna(0)
            put_vol_oi = (puts['volume'].fillna(0) / puts['openInterest'].fillna(1)).fillna(0)
            
            # Unusual activity score
            unusual_calls = (call_vol_oi > 2.0).sum()
            unusual_puts = (put_vol_oi > 2.0).sum()
            
            return unusual_calls + unusual_puts
            
        except Exception:
            return 0
    
    def calculate_dealer_positioning(self, calls, puts):
        """Calculate dealer gamma positioning"""
        
        try:
            # Net gamma exposure
            call_gamma = (calls['gamma'].fillna(0) * calls['openInterest'].fillna(0)).sum()
            put_gamma = (puts['gamma'].fillna(0) * puts['openInterest'].fillna(0)).sum()
            
            net_gamma = call_gamma - put_gamma
            
            # Normalize to -1 to 1 scale
            if abs(net_gamma) > 0:
                return max(-1, min(1, net_gamma / 10000))
            else:
                return 0
                
        except Exception:
            return 0
    
    def find_key_strikes(self, calls, puts):
        """Find strikes with high open interest"""
        
        try:
            all_strikes = []
            
            # Get top call strikes by OI
            call_strikes = calls.nlargest(3, 'openInterest')['strike'].tolist()
            put_strikes = puts.nlargest(3, 'openInterest')['strike'].tolist()
            
            all_strikes.extend(call_strikes)
            all_strikes.extend(put_strikes)
            
            return sorted(list(set(all_strikes)))
            
        except Exception:
            return []
    
    def calculate_sentiment(self, flow_data):
        """Calculate overall sentiment from flow data"""
        
        try:
            sentiment_score = 0
            
            # Put/call ratio contribution
            pc_ratio = flow_data.get('pc_ratio', 1.0)
            if pc_ratio > 1.5:
                sentiment_score -= 0.3  # Bearish
            elif pc_ratio < 0.7:
                sentiment_score += 0.3  # Bullish
            
            # Dealer positioning contribution
            dealer_pos = flow_data.get('dealer_pos', 0)
            sentiment_score += dealer_pos * 0.4
            
            # Unusual activity contribution
            unusual = flow_data.get('unusual', 0)
            if unusual > 2:
                sentiment_score += 0.2 if dealer_pos > 0 else -0.2
            
            # Convert to label
            if sentiment_score > 0.3:
                return 'BULLISH'
            elif sentiment_score < -0.3:
                return 'BEARISH'
            else:
                return 'NEUTRAL'
                
        except Exception:
            return 'NEUTRAL'
    
    def generate_futures_signals(self, flow_data):
        """Generate ES/NQ futures trading signals from options flow"""
        
        signals = {
            'ES': {'action': 'NEUTRAL', 'confidence': 0},
            'NQ': {'action': 'NEUTRAL', 'confidence': 0}
        }
        
        try:
            # ES SIGNALS
            if 'ES' in flow_data and flow_data['ES']:
                es_data = flow_data['ES']
                pc_ratio = es_data.get('put_call_ratio', 1.0)
                dealer_pos = es_data.get('dealer_positioning', 0)
                
                if pc_ratio > 1.5 and dealer_pos < -0.5:
                    signals['ES'] = {
                        'action': 'BUY',
                        'confidence': 0.85,
                        'reason': 'Extreme put buying, dealers short gamma'
                    }
                elif pc_ratio < 0.5 and dealer_pos > 0.5:
                    signals['ES'] = {
                        'action': 'SELL',
                        'confidence': 0.85,
                        'reason': 'Extreme call buying, dealers long gamma'
                    }
            
            # NQ SIGNALS
            if 'NQ' in flow_data and flow_data['NQ']:
                nq_data = flow_data['NQ']
                unusual = nq_data.get('unusual_activity', 0)
                sentiment = nq_data.get('sentiment', 'NEUTRAL')
                
                if unusual > 2.0:
                    if sentiment == 'BULLISH':
                        signals['NQ'] = {
                            'action': 'BUY',
                            'confidence': 0.80,
                            'reason': 'Unusual bullish activity detected'
                        }
                    elif sentiment == 'BEARISH':
                        signals['NQ'] = {
                            'action': 'SELL',
                            'confidence': 0.80,
                            'reason': 'Unusual bearish activity detected'
                        }
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
        
        return signals

def main():
    """Main execution function"""
    try:
        # Create analyzer
        options_analyzer = ES_NQ_OptionsFlowAnalyzer()
        
        # Analyze flow
        logger.info("Starting ES/NQ options flow analysis...")
        flow_data = options_analyzer.analyze_realtime_flow()
        
        # Save for bot consumption
        output_file = os.path.join(options_analyzer.output_dir, 'es_nq_flow.json')
        with open(output_file, 'w') as f:
            json.dump(flow_data, f, indent=2)
        
        # Log results
        if 'signals' in flow_data:
            print(f"[OPTIONS] ES Signal: {flow_data['signals']['ES']}")
            print(f"[OPTIONS] NQ Signal: {flow_data['signals']['NQ']}")
        
        logger.info(f"Options flow analysis complete. Data saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        
        # Create fallback data
        fallback_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e),
            'ES': {'sentiment': 'NEUTRAL'},
            'NQ': {'sentiment': 'NEUTRAL'},
            'signals': {
                'ES': {'action': 'NEUTRAL', 'confidence': 0},
                'NQ': {'action': 'NEUTRAL', 'confidence': 0}
            }
        }
        
        output_file = "Intelligence/data/options/es_nq_flow.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(fallback_data, f, indent=2)

if __name__ == "__main__":
    main()