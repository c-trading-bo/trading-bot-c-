#!/usr/bin/env python3
"""
Rebalancing Flows Analysis for Intelligence Pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RebalancingAnalyzer:
    def __init__(self):
        self.data_dir = "Intelligence/data/rebalancing"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def analyze_rebalancing_flows(self):
        """Analyze month-end and quarter-end rebalancing flows"""
        try:
            today = datetime.now()
            
            # Check if it's month-end or quarter-end
            is_month_end = today.day >= 25
            is_quarter_end = today.month in [3, 6, 9, 12] and is_month_end
            
            analysis = {
                'is_month_end_period': is_month_end,
                'is_quarter_end_period': is_quarter_end,
                'expected_volatility': 'elevated' if is_month_end else 'normal',
                'rebalancing_pressure': 'high' if is_quarter_end else 'moderate' if is_month_end else 'low',
                'expected_flows': 'equity_selling' if is_quarter_end else 'mixed'
            }
            
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing rebalancing flows: {e}")
            return {}

def main():
    analyzer = RebalancingAnalyzer()
    analysis = analyzer.analyze_rebalancing_flows()
    
    timestamp_str = datetime.now().strftime("%Y-%m-%d")
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'rebalancing_analysis': analysis,
        'generated_by': 'rebalancing_analyzer'
    }
    
    filepath = os.path.join(analyzer.data_dir, f"{timestamp_str}.json")
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info("Rebalancing flows analysis completed")

if __name__ == "__main__":
    main()