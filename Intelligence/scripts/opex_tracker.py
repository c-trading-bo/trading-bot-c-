#!/usr/bin/env python3
"""
OPEX Tracker for Intelligence Pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpexTracker:
    def __init__(self):
        self.data_dir = "Intelligence/data/calendar/opex"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def track_opex_dates(self):
        """Track options expiry dates and pinning risk"""
        try:
            # Calculate next OPEX dates
            today = datetime.now()
            
            # Third Friday of each month (simplified)
            next_monthly_opex = self.get_next_monthly_opex(today)
            next_quarterly_opex = self.get_next_quarterly_opex(today)
            
            analysis = {
                'next_monthly_opex': next_monthly_opex.strftime("%Y-%m-%d"),
                'days_to_monthly_opex': (next_monthly_opex - today).days,
                'next_quarterly_opex': next_quarterly_opex.strftime("%Y-%m-%d"),
                'days_to_quarterly_opex': (next_quarterly_opex - today).days,
                'pinning_risk': 'high' if (next_monthly_opex - today).days <= 3 else 'low'
            }
            return analysis
        except Exception as e:
            logger.error(f"Error tracking OPEX: {e}")
            return {}
    
    def get_next_monthly_opex(self, date):
        """Get next monthly OPEX (third Friday)"""
        # Simplified - third Friday of current or next month
        year, month = date.year, date.month
        third_friday = datetime(year, month, 15)  # Approximation
        
        if third_friday < date:
            if month == 12:
                third_friday = datetime(year + 1, 1, 15)
            else:
                third_friday = datetime(year, month + 1, 15)
        
        return third_friday
    
    def get_next_quarterly_opex(self, date):
        """Get next quarterly OPEX"""
        # March, June, September, December
        quarterly_months = [3, 6, 9, 12]
        current_month = date.month
        
        next_quarter_month = next((m for m in quarterly_months if m > current_month), quarterly_months[0])
        year = date.year if next_quarter_month > current_month else date.year + 1
        
        return datetime(year, next_quarter_month, 15)

def main():
    tracker = OpexTracker()
    analysis = tracker.track_opex_dates()
    
    timestamp_str = datetime.now().strftime("%Y-%m")
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'opex_calendar': analysis,
        'generated_by': 'opex_tracker'
    }
    
    filepath = os.path.join(tracker.data_dir, f"{timestamp_str}.json")
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info("OPEX tracking completed")

if __name__ == "__main__":
    main()