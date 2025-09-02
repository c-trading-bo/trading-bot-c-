#!/usr/bin/env python3
"""
Live Zone Tracker - Real-time zone monitoring and updates
Part of the enhanced zone detection system for institutional-grade trading
"""

import json
import asyncio
import datetime
import aiofiles
from pathlib import Path
from typing import Dict, List, Optional
import logging

class LiveZoneTracker:
    """
    Real-time zone tracking and management system
    Monitors zone interactions and updates zone effectiveness in real-time
    """
    
    def __init__(self, zones_dir: str = "Intelligence/data/zones"):
        self.zones_dir = Path(zones_dir)
        self.active_zones_file = self.zones_dir / "active_zones.json"
        self.learning_dir = self.zones_dir / "learning"
        self.logger = self._setup_logger()
        
        # Ensure directories exist
        self.zones_dir.mkdir(parents=True, exist_ok=True)
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the zone tracker"""
        logger = logging.getLogger("LiveZoneTracker")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    async def load_active_zones(self, symbol: str = "ES") -> Dict:
        """Load current active zones for a symbol"""
        try:
            zones_file = self.zones_dir / f"active_zones_{symbol}.json"
            if not zones_file.exists():
                zones_file = self.active_zones_file
                
            if zones_file.exists():
                async with aiofiles.open(zones_file, 'r') as f:
                    content = await f.read()
                    return json.loads(content)
            else:
                self.logger.warning(f"No active zones file found for {symbol}")
                return self._create_empty_zones_data(symbol)
                
        except Exception as e:
            self.logger.error(f"Error loading active zones: {e}")
            return self._create_empty_zones_data(symbol)
    
    def _create_empty_zones_data(self, symbol: str) -> Dict:
        """Create empty zones data structure"""
        return {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "symbol": symbol,
            "supply_zones": [],
            "demand_zones": [],
            "poc": 0.0,
            "current_price": 0.0,
            "nearest_supply": None,
            "nearest_demand": None,
            "statistics": {
                "total_supply_zones": 0,
                "total_demand_zones": 0,
                "active_supply": 0,
                "active_demand": 0
            },
            "key_levels": {
                "nearest_support": 0.0,
                "nearest_resistance": 0.0,
                "strongest_support": 0.0,
                "strongest_resistance": 0.0
            }
        }
    
    async def update_zone_interaction(self, symbol: str, price: float, 
                                    zone_type: str, interaction_type: str):
        """
        Update zone interaction data in real-time
        
        Args:
            symbol: Trading symbol (ES, NQ, etc.)
            price: Current price where interaction occurred
            zone_type: 'supply' or 'demand'
            interaction_type: 'test', 'hold', 'break', 'reject'
        """
        try:
            zones_data = await self.load_active_zones(symbol)
            
            # Find the relevant zone
            zones_list = zones_data.get(f"{zone_type}_zones", [])
            relevant_zone = None
            
            for zone in zones_list:
                zone_bottom = zone.get("zone_bottom", zone.get("price_level", 0))
                zone_top = zone.get("zone_top", zone.get("price_level", 0))
                
                if zone_bottom <= price <= zone_top:
                    relevant_zone = zone
                    break
            
            if relevant_zone:
                # Update zone statistics
                relevant_zone["touches"] = relevant_zone.get("touches", 0) + 1
                relevant_zone["last_test"] = datetime.datetime.utcnow().isoformat()
                
                if interaction_type == "hold":
                    relevant_zone["holds"] = relevant_zone.get("holds", 0) + 1
                elif interaction_type == "break":
                    relevant_zone["breaks"] = relevant_zone.get("breaks", 0) + 1
                    # Deactivate broken zones
                    relevant_zone["active"] = False
                
                # Update strength based on effectiveness
                if relevant_zone["touches"] > 0:
                    hold_rate = relevant_zone.get("holds", 0) / relevant_zone["touches"]
                    relevant_zone["strength"] = min(100.0, hold_rate * 100 + 
                                                  relevant_zone.get("volume", 0) / 1000)
                
                # Save updated zones
                await self._save_zones_data(symbol, zones_data)
                
                # Log interaction for learning
                await self._log_zone_interaction(symbol, price, zone_type, 
                                               interaction_type, relevant_zone)
                
                self.logger.info(f"Zone interaction recorded: {symbol} {zone_type} "
                               f"at {price} - {interaction_type}")
            
        except Exception as e:
            self.logger.error(f"Error updating zone interaction: {e}")
    
    async def _save_zones_data(self, symbol: str, zones_data: Dict):
        """Save updated zones data to file"""
        try:
            zones_file = self.zones_dir / f"active_zones_{symbol}.json"
            
            async with aiofiles.open(zones_file, 'w') as f:
                await f.write(json.dumps(zones_data, indent=2))
                
            # Also update the main active zones file if ES
            if symbol == "ES":
                async with aiofiles.open(self.active_zones_file, 'w') as f:
                    await f.write(json.dumps(zones_data, indent=2))
                    
        except Exception as e:
            self.logger.error(f"Error saving zones data: {e}")
    
    async def _log_zone_interaction(self, symbol: str, price: float, 
                                  zone_type: str, interaction_type: str, 
                                  zone_data: Dict):
        """Log zone interaction for machine learning"""
        try:
            timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            interaction_file = self.learning_dir / f"zone_interactions_{timestamp}.json"
            
            interaction_record = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "symbol": symbol,
                "price": price,
                "zone_type": zone_type,
                "interaction_type": interaction_type,
                "zone_level": zone_data.get("price_level", 0),
                "zone_strength": zone_data.get("strength", 0),
                "zone_age_hours": self._calculate_zone_age(zone_data),
                "touches_before": zone_data.get("touches", 0) - 1,
                "holds_before": zone_data.get("holds", 0),
                "breaks_before": zone_data.get("breaks", 0)
            }
            
            async with aiofiles.open(interaction_file, 'w') as f:
                await f.write(json.dumps(interaction_record, indent=2))
                
        except Exception as e:
            self.logger.error(f"Error logging zone interaction: {e}")
    
    def _calculate_zone_age(self, zone_data: Dict) -> float:
        """Calculate zone age in hours"""
        try:
            created_str = zone_data.get("created_date", "")
            if created_str:
                created_time = datetime.datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                age = datetime.datetime.utcnow() - created_time.replace(tzinfo=None)
                return age.total_seconds() / 3600  # Hours
        except:
            pass
        return 0.0
    
    async def cleanup_inactive_zones(self, symbol: str = "ES", 
                                   max_age_hours: int = 168):  # 1 week
        """Remove zones that are too old or ineffective"""
        try:
            zones_data = await self.load_active_zones(symbol)
            current_time = datetime.datetime.utcnow()
            
            for zone_type in ["supply_zones", "demand_zones"]:
                active_zones = []
                
                for zone in zones_data.get(zone_type, []):
                    # Check age
                    zone_age = self._calculate_zone_age(zone)
                    
                    # Check effectiveness (hold rate)
                    touches = zone.get("touches", 1)
                    holds = zone.get("holds", 0)
                    hold_rate = holds / max(touches, 1)
                    
                    # Keep zones that are:
                    # - Less than max_age_hours old
                    # - Have good hold rate (>50%)
                    # - Are marked as active
                    if (zone_age < max_age_hours and 
                        hold_rate > 0.5 and 
                        zone.get("active", True)):
                        active_zones.append(zone)
                
                zones_data[zone_type] = active_zones
            
            # Update statistics
            zones_data["statistics"]["active_supply"] = len(zones_data.get("supply_zones", []))
            zones_data["statistics"]["active_demand"] = len(zones_data.get("demand_zones", []))
            zones_data["timestamp"] = current_time.isoformat()
            
            await self._save_zones_data(symbol, zones_data)
            
            self.logger.info(f"Zone cleanup completed for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up zones: {e}")
    
    async def start_monitoring(self, symbols: List[str] = ["ES", "NQ"], 
                              cleanup_interval: int = 3600):  # 1 hour
        """Start continuous zone monitoring"""
        self.logger.info(f"Starting live zone monitoring for {symbols}")
        
        while True:
            try:
                for symbol in symbols:
                    await self.cleanup_inactive_zones(symbol)
                    
                await asyncio.sleep(cleanup_interval)
                
            except KeyboardInterrupt:
                self.logger.info("Zone monitoring stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying


async def main():
    """Main function for standalone execution"""
    tracker = LiveZoneTracker()
    
    # Example usage
    await tracker.update_zone_interaction("ES", 5720.0, "supply", "test")
    await tracker.update_zone_interaction("ES", 5680.0, "demand", "hold")
    await tracker.cleanup_inactive_zones("ES")
    
    print("Live zone tracker example completed")


if __name__ == "__main__":
    asyncio.run(main())