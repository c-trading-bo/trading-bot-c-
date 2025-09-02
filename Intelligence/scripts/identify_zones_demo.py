#!/usr/bin/env python3
"""
Demo Supply/Demand Zones Identifier for Intelligence Pipeline
Creates realistic zones for testing without requiring live market data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DemoZonesIdentifier:
    def __init__(self):
        self.data_dir = "Intelligence/data/zones"
        self.feedback_dir = "Intelligence/data/zones/feedback"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.feedback_dir, exist_ok=True)
        
    def generate_realistic_zones(self, symbol="ES", current_price=4500.0):
        """Generate realistic supply and demand zones based on current market conditions"""
        
        # ES/NQ-specific parameters
        if symbol.upper() in ['ES', 'ES=F']:
            tick_size = 0.25
            point_value = 50
            typical_range = 50.0  # 50 points typical daily range
        elif symbol.upper() in ['NQ', 'NQ=F']:
            tick_size = 0.25
            point_value = 20
            typical_range = 200.0  # 200 points typical daily range
            current_price = current_price * 3.6  # Scale up for NQ
        else:
            tick_size = 0.25
            point_value = 50
            typical_range = 50.0
        
        zones = []
        
        # Generate supply zones (resistance levels above current price)
        supply_levels = [
            current_price + typical_range * 0.3,   # Nearby resistance
            current_price + typical_range * 0.6,   # Medium resistance
            current_price + typical_range * 0.9,   # Strong resistance
            current_price + typical_range * 1.2,   # Major resistance
        ]
        
        for i, level in enumerate(supply_levels):
            # Round to tick size
            level = round(level / tick_size) * tick_size
            
            # Zone strength decreases with distance but varies
            base_strength = 85 - (i * 15) + np.random.randint(-10, 10)
            strength = max(40, min(95, base_strength))
            
            # Zone width based on volatility and strength
            width_pct = 0.002 + (i * 0.001)  # 0.2% to 0.5%
            width = level * width_pct
            
            zone = {
                'type': 'supply',
                'level': round(level, 2),
                'upper': round(level + width/2, 2),
                'lower': round(level - width/2, 2),
                'strength': strength,
                'age_days': np.random.randint(1, 14),
                'touch_count': max(1, np.random.poisson(2)),
                'last_test': (datetime.now() - timedelta(days=np.random.randint(0, 7))).isoformat(),
                'volume_confirmation': strength > 70,
                'creation_time': (datetime.now() - timedelta(days=np.random.randint(1, 30))).isoformat(),
                'source': 'swing_high'
            }
            zones.append(zone)
        
        # Generate demand zones (support levels below current price)
        demand_levels = [
            current_price - typical_range * 0.3,   # Nearby support
            current_price - typical_range * 0.6,   # Medium support  
            current_price - typical_range * 0.9,   # Strong support
            current_price - typical_range * 1.2,   # Major support
        ]
        
        for i, level in enumerate(demand_levels):
            # Round to tick size
            level = round(level / tick_size) * tick_size
            
            # Zone strength decreases with distance but varies
            base_strength = 90 - (i * 12) + np.random.randint(-8, 8)
            strength = max(45, min(95, base_strength))
            
            # Zone width based on volatility and strength
            width_pct = 0.002 + (i * 0.001)  # 0.2% to 0.5%
            width = level * width_pct
            
            zone = {
                'type': 'demand',
                'level': round(level, 2),
                'upper': round(level + width/2, 2),
                'lower': round(level - width/2, 2),
                'strength': strength,
                'age_days': np.random.randint(1, 12),
                'touch_count': max(1, np.random.poisson(3)),
                'last_test': (datetime.now() - timedelta(days=np.random.randint(0, 5))).isoformat(),
                'volume_confirmation': strength > 75,
                'creation_time': (datetime.now() - timedelta(days=np.random.randint(1, 25))).isoformat(),
                'source': 'swing_low'
            }
            zones.append(zone)
        
        # Add some intraday zones (closer to current price)
        intraday_zones = []
        
        # Intraday supply zone
        intraday_supply = current_price + (typical_range * 0.15)
        intraday_supply = round(intraday_supply / tick_size) * tick_size
        width = intraday_supply * 0.001
        
        intraday_zones.append({
            'type': 'supply',
            'level': round(intraday_supply, 2),
            'upper': round(intraday_supply + width/2, 2),
            'lower': round(intraday_supply - width/2, 2),
            'strength': np.random.randint(60, 80),
            'age_days': 0,
            'touch_count': 1,
            'last_test': datetime.now().isoformat(),
            'volume_confirmation': True,
            'creation_time': (datetime.now() - timedelta(hours=np.random.randint(1, 8))).isoformat(),
            'source': 'intraday_high'
        })
        
        # Intraday demand zone
        intraday_demand = current_price - (typical_range * 0.15)
        intraday_demand = round(intraday_demand / tick_size) * tick_size
        width = intraday_demand * 0.001
        
        intraday_zones.append({
            'type': 'demand',
            'level': round(intraday_demand, 2),
            'upper': round(intraday_demand + width/2, 2),
            'lower': round(intraday_demand - width/2, 2),
            'strength': np.random.randint(65, 85),
            'age_days': 0,
            'touch_count': 1,
            'last_test': datetime.now().isoformat(),
            'volume_confirmation': True,
            'creation_time': (datetime.now() - timedelta(hours=np.random.randint(1, 6))).isoformat(),
            'source': 'intraday_low'
        })
        
        zones.extend(intraday_zones)
        
        # Sort zones by strength (strongest first)
        zones.sort(key=lambda x: x['strength'], reverse=True)
        
        return zones
    
    def calculate_volume_profile(self, symbol="ES"):
        """Generate realistic volume profile data"""
        
        if symbol.upper() in ['ES', 'ES=F']:
            current_price = 4500.0
            typical_range = 50.0
        elif symbol.upper() in ['NQ', 'NQ=F']:
            current_price = 16200.0
            typical_range = 200.0
        else:
            current_price = 4500.0
            typical_range = 50.0
        
        # Generate Point of Control (POC) and Value Area
        poc = current_price + np.random.uniform(-typical_range*0.1, typical_range*0.1)
        
        # Value area typically covers 70% of volume
        value_area_width = typical_range * 0.4
        value_area_high = poc + value_area_width * 0.6
        value_area_low = poc - value_area_width * 0.4
        
        # High Volume Nodes (HVN) and Low Volume Nodes (LVN)
        hvn_levels = []
        lvn_levels = []
        
        # Generate 3-5 HVN levels around POC
        for i in range(np.random.randint(3, 6)):
            offset = np.random.uniform(-typical_range*0.3, typical_range*0.3)
            hvn_level = poc + offset
            volume_intensity = np.random.randint(80, 100)
            
            hvn_levels.append({
                'price': round(hvn_level, 2),
                'volume_intensity': volume_intensity,
                'type': 'hvn'
            })
        
        # Generate 2-4 LVN levels (potential breakout points)
        for i in range(np.random.randint(2, 5)):
            # LVNs tend to be further from POC
            offset = np.random.uniform(-typical_range*0.8, typical_range*0.8)
            if abs(offset) < typical_range * 0.2:  # Avoid area near POC
                offset = typical_range * 0.4 * (1 if offset > 0 else -1)
            
            lvn_level = poc + offset
            volume_intensity = np.random.randint(10, 30)
            
            lvn_levels.append({
                'price': round(lvn_level, 2),
                'volume_intensity': volume_intensity,
                'type': 'lvn'
            })
        
        volume_profile = {
            'poc': round(poc, 2),
            'value_area_high': round(value_area_high, 2),
            'value_area_low': round(value_area_low, 2),
            'hvn_levels': hvn_levels,
            'lvn_levels': lvn_levels,
            'generated_at': datetime.now().isoformat()
        }
        
        return volume_profile
    
    def create_zones_analysis(self, symbol="ES"):
        """Create comprehensive zones analysis"""
        
        # Current price based on symbol
        if symbol.upper() in ['ES', 'ES=F']:
            current_price = 4500.0 + np.random.uniform(-25, 25)  # ES around 4500
        elif symbol.upper() in ['NQ', 'NQ=F']:
            current_price = 16200.0 + np.random.uniform(-100, 100)  # NQ around 16200
        else:
            current_price = 4500.0 + np.random.uniform(-25, 25)
        
        # Generate zones
        zones = self.generate_realistic_zones(symbol, current_price)
        
        # Generate volume profile
        volume_profile = self.calculate_volume_profile(symbol)
        
        # Filter for most relevant zones (within 2% of current price)
        price_range = current_price * 0.02
        relevant_zones = [
            zone for zone in zones 
            if abs(zone['level'] - current_price) <= price_range * 2.5
        ]
        
        # Create analysis summary
        supply_zones = [z for z in relevant_zones if z['type'] == 'supply']
        demand_zones = [z for z in relevant_zones if z['type'] == 'demand']
        
        analysis = {
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'analysis_time': datetime.now().isoformat(),
            'all_zones': zones,
            'relevant_zones': relevant_zones,
            'supply_zones': supply_zones,
            'demand_zones': demand_zones,
            'volume_profile': volume_profile,
            'key_levels': {
                'nearest_support': min([z['level'] for z in demand_zones if z['level'] < current_price], default=current_price - 20),
                'nearest_resistance': min([z['level'] for z in supply_zones if z['level'] > current_price], default=current_price + 20),
                'strongest_support': max(demand_zones, key=lambda x: x['strength'])['level'] if demand_zones else current_price - 30,
                'strongest_resistance': max(supply_zones, key=lambda x: x['strength'])['level'] if supply_zones else current_price + 30
            },
            'zone_summary': {
                'total_zones': len(zones),
                'supply_count': len(supply_zones),
                'demand_count': len(demand_zones),
                'average_strength': round(np.mean([z['strength'] for z in relevant_zones]), 1) if relevant_zones else 0,
                'high_strength_zones': len([z for z in relevant_zones if z['strength'] >= 80])
            }
        }
        
        return analysis
    
    def save_zones_data(self, analysis):
        """Save zones analysis to files"""
        symbol = analysis['symbol']
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed analysis
        zones_file = os.path.join(self.data_dir, f"zones_{symbol}_{timestamp_str}.json")
        with open(zones_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Save active zones summary for bot consumption
        active_zones = {
            'symbol': symbol,
            'current_price': analysis['current_price'],
            'supply_zones': [
                {
                    'price': z['level'],
                    'range': [z['lower'], z['upper']],
                    'strength': z['strength'],
                    'last_test': z['last_test'],
                    'touch_count': z['touch_count']
                }
                for z in analysis['supply_zones'][:5]  # Top 5 supply zones
            ],
            'demand_zones': [
                {
                    'price': z['level'],
                    'range': [z['lower'], z['upper']],
                    'strength': z['strength'],
                    'last_test': z['last_test'],
                    'touch_count': z['touch_count']
                }
                for z in analysis['demand_zones'][:5]  # Top 5 demand zones
            ],
            'poc': analysis['volume_profile']['poc'],
            'value_area': {
                'high': analysis['volume_profile']['value_area_high'],
                'low': analysis['volume_profile']['value_area_low']
            },
            'key_levels': analysis['key_levels'],
            'generated_at': analysis['analysis_time'],
            'next_update': (datetime.now() + timedelta(hours=4)).isoformat()
        }
        
        # Save active zones for bot
        active_file = os.path.join(self.data_dir, f"active_zones_{symbol}.json")
        with open(active_file, 'w') as f:
            json.dump(active_zones, f, indent=2)
        
        # Update latest zones for the intelligence service
        latest_file = os.path.join(self.data_dir, "latest_zones.json")
        with open(latest_file, 'w') as f:
            json.dump(active_zones, f, indent=2)
        
        logger.info(f"Saved zones for {symbol}: {len(analysis['supply_zones'])} supply, "
                   f"{len(analysis['demand_zones'])} demand zones")
        
        return zones_file, active_file

def main():
    """Main execution function"""
    import sys
    
    identifier = DemoZonesIdentifier()
    
    # Support multiple symbols
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ['ES', 'NQ']
    
    for symbol in symbols:
        logger.info(f"Generating zones for {symbol}...")
        
        try:
            analysis = identifier.create_zones_analysis(symbol)
            zones_file, active_file = identifier.save_zones_data(analysis)
            
            print(f"\n{symbol} Zones Analysis:")
            print(f"Current Price: {analysis['current_price']}")
            print(f"Nearest Support: {analysis['key_levels']['nearest_support']}")
            print(f"Nearest Resistance: {analysis['key_levels']['nearest_resistance']}")
            print(f"POC: {analysis['volume_profile']['poc']}")
            print(f"Value Area: {analysis['volume_profile']['value_area_low']} - {analysis['volume_profile']['value_area_high']}")
            print(f"Total Zones: {analysis['zone_summary']['total_zones']}")
            print(f"High Strength Zones: {analysis['zone_summary']['high_strength_zones']}")
            
        except Exception as e:
            logger.error(f"Error generating zones for {symbol}: {e}")

if __name__ == "__main__":
    main()