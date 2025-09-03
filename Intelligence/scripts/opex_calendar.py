#!/usr/bin/env python3
"""
OPEX Calendar Monitor for ES/NQ Trading Intelligence
Tracks options expiration events and their market impact
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import logging
import calendar

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OPEXCalendarMonitor:
    def __init__(self):
        self.data_dir = "Intelligence/data/opex"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_opex_dates(self, year=None, month=None):
        """Calculate OPEX dates (3rd Friday of each month)"""
        if year is None:
            year = datetime.now().year
        if month is None:
            month = datetime.now().month
            
        # Find 3rd Friday of the month
        first_day = datetime(year, month, 1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)
        
        return third_friday
    
    def analyze_opex_impact(self):
        """Analyze upcoming OPEX events and their market impact"""
        try:
            now = datetime.now()
            current_month_opex = self.get_opex_dates()
            next_month_opex = self.get_opex_dates(month=(now.month % 12) + 1)
            
            # Calculate days to next OPEX
            if current_month_opex >= now.date():
                next_opex = current_month_opex
            else:
                next_opex = next_month_opex
            
            days_to_opex = (next_opex.date() - now.date()).days
            
            # Quarterly OPEX (March, June, September, December)
            is_quarterly = next_opex.month in [3, 6, 9, 12]
            
            # OPEX week volatility patterns
            if days_to_opex <= 7:
                opex_phase = 'OPEX_WEEK'
                volatility_expectation = 'HIGH' if is_quarterly else 'ELEVATED'
                market_impact = 'PINNING' if days_to_opex <= 2 else 'VOLATILITY'
            elif days_to_opex <= 14:
                opex_phase = 'PRE_OPEX'
                volatility_expectation = 'MODERATE'
                market_impact = 'POSITIONING'
            else:
                opex_phase = 'NORMAL'
                volatility_expectation = 'NORMAL'
                market_impact = 'MINIMAL'
            
            # Key strike levels (simulated based on current market)
            # In production, this would fetch actual options data
            estimated_spy_price = 450  # Would fetch real SPY price
            key_strikes = [
                estimated_spy_price - 10,
                estimated_spy_price - 5,
                estimated_spy_price,
                estimated_spy_price + 5,
                estimated_spy_price + 10
            ]
            
            analysis = {
                'next_opex_date': next_opex.strftime('%Y-%m-%d'),
                'days_to_opex': days_to_opex,
                'is_quarterly_opex': is_quarterly,
                'opex_phase': opex_phase,
                'volatility_expectation': volatility_expectation,
                'market_impact': market_impact,
                'key_strike_levels': key_strikes,
                'trading_implications': {
                    'expected_volatility': 'HIGH' if days_to_opex <= 3 else 'MODERATE',
                    'pin_risk': days_to_opex <= 2,
                    'gamma_squeeze_risk': days_to_opex <= 1 and is_quarterly,
                    'recommended_strategy': self.get_opex_strategy(days_to_opex, is_quarterly)
                },
                'trade_recommendation': {
                    'position_size_multiplier': self.get_opex_position_sizing(days_to_opex, is_quarterly),
                    'stop_loss_multiplier': self.get_opex_stop_sizing(days_to_opex, is_quarterly),
                    'timeframe_bias': 'SHORT_TERM' if days_to_opex <= 5 else 'NORMAL'
                }
            }
            
            logger.info(f"OPEX Analysis: {days_to_opex} days to {'quarterly' if is_quarterly else 'monthly'} OPEX")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing OPEX calendar: {e}")
            return {}
    
    def get_opex_strategy(self, days_to_opex, is_quarterly):
        """Recommend trading strategy based on OPEX timing"""
        if days_to_opex <= 1:
            return 'AVOID_NEW_POSITIONS' if is_quarterly else 'SCALP_ONLY'
        elif days_to_opex <= 3:
            return 'REDUCED_SIZE' if is_quarterly else 'MOMENTUM_FADE'
        elif days_to_opex <= 7:
            return 'VOLATILITY_EXPANSION'
        else:
            return 'NORMAL_STRATEGIES'
    
    def get_opex_position_sizing(self, days_to_opex, is_quarterly):
        """Calculate position size multiplier based on OPEX proximity"""
        if days_to_opex <= 1 and is_quarterly:
            return 0.3  # Very small positions
        elif days_to_opex <= 3:
            return 0.6  # Reduced size
        elif days_to_opex <= 7:
            return 0.8  # Slightly reduced
        else:
            return 1.0  # Normal size
    
    def get_opex_stop_sizing(self, days_to_opex, is_quarterly):
        """Calculate stop loss multiplier based on OPEX volatility"""
        if days_to_opex <= 1 and is_quarterly:
            return 2.5  # Very wide stops
        elif days_to_opex <= 3:
            return 1.8  # Wider stops
        elif days_to_opex <= 7:
            return 1.4  # Slightly wider
        else:
            return 1.0  # Normal stops
    
    def save_analysis(self, analysis):
        """Save OPEX analysis to file"""
        try:
            output_file = os.path.join(self.data_dir, 'latest_opex_analysis.json')
            
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            logger.info(f"Saved OPEX analysis to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving OPEX analysis: {e}")

def main():
    """Main OPEX calendar monitoring workflow"""
    logger.info("Starting OPEX Calendar monitoring...")
    
    monitor = OPEXCalendarMonitor()
    
    # Analyze OPEX impact
    analysis = monitor.analyze_opex_impact()
    if analysis:
        monitor.save_analysis(analysis)
        logger.info("OPEX calendar analysis completed successfully")
    else:
        logger.error("Failed to analyze OPEX calendar")

if __name__ == "__main__":
    main()
