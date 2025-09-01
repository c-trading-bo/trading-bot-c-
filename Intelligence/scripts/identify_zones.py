#!/usr/bin/env python3
"""
Supply/Demand Zones Identifier for Intelligence Pipeline
Identifies key supply and demand zones with feedback integration
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

class ZonesIdentifier:
    def __init__(self):
        self.data_dir = "Intelligence/data/zones/daily"
        self.feedback_dir = "Intelligence/data/zones/feedback"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.feedback_dir, exist_ok=True)
        self.symbols = ['ES=F', 'NQ=F', 'YM=F', 'RTY=F']
        
    def identify_zones(self, symbol, period="5d"):
        """Identify supply and demand zones"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval="5m")
            
            if data.empty:
                return []
            
            zones = []
            
            # Identify swing highs and lows
            swing_highs = self.find_swing_points(data['High'], lookback=5, point_type='high')
            swing_lows = self.find_swing_points(data['Low'], lookback=5, point_type='low')
            
            # Create supply zones from swing highs
            for idx, price in swing_highs:
                zone = self.create_zone(data, idx, price, 'supply')
                if zone:
                    zones.append(zone)
            
            # Create demand zones from swing lows
            for idx, price in swing_lows:
                zone = self.create_zone(data, idx, price, 'demand')
                if zone:
                    zones.append(zone)
            
            # Filter and rank zones
            zones = self.filter_zones(zones, data['Close'].iloc[-1])
            zones = self.rank_zones(zones)
            
            return zones[:10]  # Return top 10 zones
            
        except Exception as e:
            logger.error(f"Error identifying zones for {symbol}: {e}")
            return []
    
    def find_swing_points(self, series, lookback=5, point_type='high'):
        """Find swing highs or lows"""
        points = []
        
        for i in range(lookback, len(series) - lookback):
            if point_type == 'high':
                if all(series.iloc[i] >= series.iloc[i-j] for j in range(1, lookback+1)) and \
                   all(series.iloc[i] >= series.iloc[i+j] for j in range(1, lookback+1)):
                    points.append((i, series.iloc[i]))
            else:  # low
                if all(series.iloc[i] <= series.iloc[i-j] for j in range(1, lookback+1)) and \
                   all(series.iloc[i] <= series.iloc[i+j] for j in range(1, lookback+1)):
                    points.append((i, series.iloc[i]))
        
        return points
    
    def create_zone(self, data, idx, price, zone_type):
        """Create a supply or demand zone"""
        try:
            # Calculate zone boundaries
            if zone_type == 'supply':
                upper = price
                lower = price * 0.998  # 0.2% zone width
            else:  # demand
                lower = price
                upper = price * 1.002  # 0.2% zone width
            
            # Calculate zone strength
            volume_strength = data['Volume'].iloc[max(0, idx-5):idx+5].mean()
            touch_count = self.count_touches(data, upper, lower)
            
            zone = {
                'type': zone_type,
                'upper': round(upper, 2),
                'lower': round(lower, 2),
                'midpoint': round((upper + lower) / 2, 2),
                'timestamp': data.index[idx].isoformat(),
                'volume_strength': int(volume_strength),
                'touch_count': touch_count,
                'strength_score': self.calculate_strength_score(volume_strength, touch_count),
                'status': 'untested'
            }
            
            return zone
            
        except Exception as e:
            logger.error(f"Error creating zone: {e}")
            return None
    
    def count_touches(self, data, upper, lower):
        """Count how many times price touched the zone"""
        touches = 0
        for i in range(len(data)):
            if lower <= data['Low'].iloc[i] <= upper or lower <= data['High'].iloc[i] <= upper:
                touches += 1
        return touches
    
    def calculate_strength_score(self, volume_strength, touch_count):
        """Calculate zone strength score"""
        # Normalize volume (simplified)
        volume_score = min(volume_strength / 1000, 10)  # Cap at 10
        
        # Touch count scoring (more touches = stronger zone, but diminishing returns)
        touch_score = min(touch_count * 2, 10)
        
        return round((volume_score + touch_score) / 2, 1)
    
    def filter_zones(self, zones, current_price):
        """Filter zones based on relevance"""
        relevant_zones = []
        
        for zone in zones:
            # Keep zones within 5% of current price
            distance = abs(zone['midpoint'] - current_price) / current_price
            if distance <= 0.05:
                zone['distance_percent'] = round(distance * 100, 2)
                relevant_zones.append(zone)
        
        return relevant_zones
    
    def rank_zones(self, zones):
        """Rank zones by strength and proximity"""
        for zone in zones:
            # Combine strength score and proximity (closer = better)
            proximity_score = 10 - zone['distance_percent']  # Closer gets higher score
            zone['total_score'] = zone['strength_score'] + proximity_score
        
        return sorted(zones, key=lambda x: x['total_score'], reverse=True)
    
    def load_feedback(self, symbol):
        """Load feedback from previous zone tests"""
        try:
            feedback_file = os.path.join(self.feedback_dir, f"{symbol}_feedback.json")
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception:
            return {}
    
    def integrate_feedback(self, zones, feedback):
        """Integrate feedback into zone analysis"""
        for zone in zones:
            zone_key = f"{zone['type']}_{zone['midpoint']}"
            if zone_key in feedback:
                fb = feedback[zone_key]
                zone['success_rate'] = fb.get('success_rate', 0)
                zone['test_count'] = fb.get('test_count', 0)
                zone['last_test'] = fb.get('last_test', None)
    
    def save_zones(self, symbol, zones):
        """Save identified zones"""
        try:
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M")
            
            output_data = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'zones': zones,
                'zone_count': len(zones),
                'generated_by': 'zones_identifier'
            }
            
            # Save timestamped file
            filename = f"{symbol}_{timestamp_str}.json"
            filepath = os.path.join(self.data_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving zones: {e}")
            return None

def main():
    """Main execution function"""
    logger.info("Starting zones identification...")
    
    identifier = ZonesIdentifier()
    
    all_zones = {}
    
    for symbol in identifier.symbols:
        logger.info(f"Analyzing zones for {symbol}")
        
        # Identify zones
        zones = identifier.identify_zones(symbol)
        
        # Load feedback
        feedback = identifier.load_feedback(symbol)
        
        # Integrate feedback
        identifier.integrate_feedback(zones, feedback)
        
        # Save zones
        filepath = identifier.save_zones(symbol, zones)
        
        if filepath:
            all_zones[symbol] = len(zones)
            logger.info(f"Zones identified for {symbol}: {len(zones)} zones")
    
    logger.info(f"Zones identification completed: {sum(all_zones.values())} total zones across {len(all_zones)} symbols")

if __name__ == "__main__":
    main()