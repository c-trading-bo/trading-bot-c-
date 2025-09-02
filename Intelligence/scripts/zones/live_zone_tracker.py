# File: Intelligence/scripts/zones/live_zone_tracker.py
import asyncio
import json
import os
from datetime import datetime
import numpy as np

class LiveZoneTracker:
    """Tracks zone interactions in real-time and learns from them"""
    
    def __init__(self):
        self.zones = self.load_zones()
        self.current_price = 0
        self.zone_interactions = []
        self.learning_data = []
        
    def load_zones(self):
        """Load active zones from file"""
        zone_file = "Intelligence/data/zones/active_zones.json"
        if os.path.exists(zone_file):
            with open(zone_file, 'r') as f:
                return json.load(f)
        return None
    
    def update_price(self, price, volume=None):
        """Update current price and check for zone interactions"""
        
        self.current_price = price
        interaction = None
        
        # Check supply zones
        for zone in self.zones.get('supply_zones', []):
            if zone['zone_bottom'] <= price <= zone['zone_top']:
                interaction = self.process_zone_touch(zone, 'supply', price, volume)
                
        # Check demand zones
        for zone in self.zones.get('demand_zones', []):
            if zone['zone_bottom'] <= price <= zone['zone_top']:
                interaction = self.process_zone_touch(zone, 'demand', price, volume)
        
        if interaction:
            self.zone_interactions.append(interaction)
            self.learn_from_interaction(interaction)
            
        return interaction
    
    def process_zone_touch(self, zone, zone_type, price, volume):
        """Process when price touches a zone"""
        
        interaction = {
            'timestamp': datetime.utcnow().isoformat(),
            'zone_type': zone_type,
            'zone_price': zone['price_level'],
            'touch_price': price,
            'zone_strength': zone['strength'],
            'volume': volume,
            'prediction': None,
            'result': None  # Will be filled after price moves away
        }
        
        # Make prediction based on zone strength
        if zone['strength'] > 70:
            if zone_type == 'supply':
                interaction['prediction'] = 'strong_rejection_down'
            else:
                interaction['prediction'] = 'strong_bounce_up'
        elif zone['strength'] > 40:
            if zone_type == 'supply':
                interaction['prediction'] = 'moderate_rejection'
            else:
                interaction['prediction'] = 'moderate_bounce'
        else:
            interaction['prediction'] = 'likely_break'
        
        print(f"[ZONE TOUCH] {zone_type.upper()} zone at {zone['price_level']:.2f}, "
              f"Strength: {zone['strength']:.0f}, Prediction: {interaction['prediction']}")
        
        return interaction
    
    def learn_from_interaction(self, interaction):
        """Learn from zone interaction outcomes"""
        
        # Wait for price to move away from zone
        # This would be called later when we know the outcome
        
        learning_sample = {
            'zone_type': interaction['zone_type'],
            'zone_strength': interaction['zone_strength'],
            'volume_at_touch': interaction['volume'],
            'time_of_day': datetime.fromisoformat(interaction['timestamp']).hour,
            'prediction': interaction['prediction'],
            'actual_result': None,  # Filled when known
            'prediction_correct': None
        }
        
        self.learning_data.append(learning_sample)
        
        # Update zone strength based on outcome
        if len(self.learning_data) > 10:
            self.update_zone_strengths()
    
    def update_zone_strengths(self):
        """Dynamically update zone strengths based on recent performance"""
        
        recent_interactions = self.learning_data[-20:]  # Last 20 interactions
        
        for zone_type in ['supply', 'demand']:
            zones = self.zones.get(f'{zone_type}_zones', [])
            
            for zone in zones:
                # Find interactions with this zone
                zone_interactions = [i for i in recent_interactions 
                                    if abs(i.get('zone_price', 0) - zone['price_level']) < 1]
                
                if zone_interactions:
                    # Calculate success rate
                    successful = sum(1 for i in zone_interactions if i.get('prediction_correct'))
                    success_rate = successful / len(zone_interactions)
                    
                    # Adjust strength
                    if success_rate > 0.7:
                        zone['strength'] = min(100, zone['strength'] * 1.1)
                    elif success_rate < 0.3:
                        zone['strength'] = max(10, zone['strength'] * 0.9)
                    
                    print(f"[LEARNING] Updated {zone_type} zone at {zone['price_level']:.2f}, "
                          f"New strength: {zone['strength']:.0f}")
    
    def save_learning_data(self):
        """Save learning data for model training"""
        
        output_dir = "Intelligence/data/zones/learning"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        with open(f"{output_dir}/zone_learning_{timestamp}.json", 'w') as f:
            json.dump({
                'interactions': self.zone_interactions,
                'learning_samples': self.learning_data,
                'zone_performance': self.calculate_zone_performance()
            }, f, indent=2)
    
    def calculate_zone_performance(self):
        """Calculate performance metrics for each zone"""
        
        performance = {}
        
        for zone_type in ['supply', 'demand']:
            zones = self.zones.get(f'{zone_type}_zones', [])
            
            for zone in zones:
                zone_key = f"{zone_type}_{zone['price_level']:.0f}"
                
                # Find all interactions with this zone
                interactions = [i for i in self.zone_interactions 
                              if abs(i.get('zone_price', 0) - zone['price_level']) < 1]
                
                if interactions:
                    performance[zone_key] = {
                        'price_level': zone['price_level'],
                        'total_touches': len(interactions),
                        'holds': sum(1 for i in interactions if i.get('result') == 'held'),
                        'breaks': sum(1 for i in interactions if i.get('result') == 'broken'),
                        'hold_rate': 0,
                        'avg_reaction_size': 0
                    }
                    
                    if performance[zone_key]['total_touches'] > 0:
                        performance[zone_key]['hold_rate'] = (
                            performance[zone_key]['holds'] / 
                            performance[zone_key]['total_touches']
                        )
        
        return performance