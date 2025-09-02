#!/usr/bin/env python3
"""
Enhanced signal generation that integrates news, zones, and market data.
Generates intelligence signals for the C# trading bot.
"""

import json
import os
from datetime import datetime, timedelta
import logging
import numpy as np
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedSignalGenerator:
    def __init__(self):
        self.base_path = "Intelligence/data"
        
    def generate_market_intelligence(self) -> Dict[str, Any]:
        """Generate comprehensive market intelligence for trading bot"""
        
        logger.info("Starting enhanced signal generation...")
        
        # Load latest news data
        news_data = self.load_latest_news()
        
        # Load zone data  
        zone_data = self.load_latest_zones()
        
        # Generate market regime
        regime = self.determine_market_regime(news_data, zone_data)
        
        # Calculate confidence based on data quality
        confidence = self.calculate_model_confidence(news_data, zone_data)
        
        # Determine primary bias
        bias = self.determine_market_bias(news_data, zone_data)
        
        # Check for major events
        events = self.check_major_events(news_data)
        
        # Generate final intelligence
        intelligence = {
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "regime": regime,
            "newsIntensity": news_data.get('news_intensity', 30.0) if news_data else 30.0,
            "isCpiDay": events.get('cpi', False),
            "isFomcDay": events.get('fomc', False),
            "modelConfidence": confidence,
            "primaryBias": bias,
            "setups": self.generate_trade_setups(regime, bias, confidence, events),
            "newsDetails": {
                "sentiment": news_data.get('avg_sentiment', 0.0) if news_data else 0.0,
                "volatilityScore": news_data.get('volatility_score', 0) if news_data else 0,
                "articleCount": news_data.get('article_count', 0) if news_data else 0,
                "newsConfidence": min(95, max(20, confidence * 100 + 15)),
                "earningsSeason": events.get('earnings', False),
                "fedSpeak": events.get('fomc', False)
            },
            "generatedAt": datetime.utcnow().isoformat(),
            "version": "1.0"
        }
        
        # Save intelligence
        self.save_intelligence(intelligence)
        
        logger.info(f"Generated intelligence: {regime} regime, {confidence:.2f} confidence, {bias} bias")
        
        return intelligence
    
    def load_latest_news(self) -> Optional[Dict[str, Any]]:
        """Load latest news analysis"""
        try:
            news_file = os.path.join(self.base_path, "news", "latest.json")
            if os.path.exists(news_file):
                with open(news_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load news data: {e}")
        return None
    
    def load_latest_zones(self) -> Optional[Dict[str, Any]]:
        """Load latest zone data"""
        try:
            zones_file = os.path.join(self.base_path, "zones", "latest_zones.json")
            if os.path.exists(zones_file):
                with open(zones_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load zone data: {e}")
        return None
    
    def determine_market_regime(self, news_data: Optional[Dict], zone_data: Optional[Dict]) -> str:
        """Determine current market regime"""
        
        # Default to ranging if no data
        if not news_data and not zone_data:
            return "Ranging"
        
        volatility_score = 0
        trend_score = 0
        
        # News-based regime detection
        if news_data:
            news_intensity = news_data.get('news_intensity', 30)
            volatility_events = news_data.get('volatility_score', 0)
            sentiment = abs(news_data.get('avg_sentiment', 0))
            
            # High news intensity suggests volatility
            if news_intensity > 70:
                volatility_score += 30
            elif news_intensity > 50:
                volatility_score += 15
            
            # Major events suggest volatility
            volatility_score += volatility_events * 10
            
            # Strong sentiment suggests trending
            if sentiment > 0.4:
                trend_score += 25
            elif sentiment > 0.2:
                trend_score += 10
        
        # Zone-based regime detection
        if zone_data:
            current_price = zone_data.get('current_price', 0)
            poc = zone_data.get('poc', 0)
            value_area = zone_data.get('value_area', {})
            
            if current_price and poc:
                # Price far from POC suggests trending
                poc_distance = abs(current_price - poc) / current_price
                if poc_distance > 0.015:  # > 1.5% from POC
                    trend_score += 20
                elif poc_distance > 0.008:  # > 0.8% from POC
                    trend_score += 10
                
                # Price outside value area suggests volatility/trending
                va_high = value_area.get('high', 0)
                va_low = value_area.get('low', 0)
                if va_high and va_low:
                    if current_price > va_high or current_price < va_low:
                        if poc_distance > 0.01:
                            trend_score += 15
                        else:
                            volatility_score += 15
        
        # Check for specific time-based patterns
        now = datetime.utcnow()
        hour = now.hour
        
        # Market open/close times tend to be more volatile
        if hour in [13, 14, 20, 21]:  # Around market open/close UTC
            volatility_score += 10
        
        # Determine regime
        if volatility_score > 40:
            return "Volatile"
        elif trend_score > 30:
            return "Trending"
        else:
            return "Ranging"
    
    def calculate_model_confidence(self, news_data: Optional[Dict], zone_data: Optional[Dict]) -> float:
        """Calculate model confidence based on data quality and consistency"""
        
        base_confidence = 0.3  # Start with 30% confidence
        
        # News data quality
        if news_data:
            article_count = news_data.get('article_count', 0)
            
            # More articles = higher confidence
            if article_count >= 20:
                base_confidence += 0.25
            elif article_count >= 10:
                base_confidence += 0.15
            elif article_count >= 5:
                base_confidence += 0.1
            
            # Consistent sentiment = higher confidence
            sentiment = abs(news_data.get('avg_sentiment', 0))
            if sentiment > 0.3:
                base_confidence += 0.2
            elif sentiment > 0.15:
                base_confidence += 0.1
            
            # Recent data = higher confidence
            try:
                timestamp = datetime.fromisoformat(news_data.get('timestamp', '').replace('Z', '+00:00'))
                age_hours = (datetime.utcnow() - timestamp.replace(tzinfo=None)).total_seconds() / 3600
                if age_hours < 1:
                    base_confidence += 0.15
                elif age_hours < 4:
                    base_confidence += 0.1
            except:
                pass
        
        # Zone data quality
        if zone_data:
            supply_zones = zone_data.get('supply_zones', [])
            demand_zones = zone_data.get('demand_zones', [])
            
            # More zones = higher confidence
            total_zones = len(supply_zones) + len(demand_zones)
            if total_zones >= 8:
                base_confidence += 0.15
            elif total_zones >= 5:
                base_confidence += 0.1
            
            # Strong zones = higher confidence
            if supply_zones or demand_zones:
                avg_strength = 0
                zone_count = 0
                
                for zone in supply_zones + demand_zones:
                    strength = zone.get('strength', 50)
                    avg_strength += strength
                    zone_count += 1
                
                if zone_count > 0:
                    avg_strength /= zone_count
                    if avg_strength > 80:
                        base_confidence += 0.2
                    elif avg_strength > 65:
                        base_confidence += 0.1
        
        # Cap confidence at 95%
        return min(0.95, base_confidence)
    
    def determine_market_bias(self, news_data: Optional[Dict], zone_data: Optional[Dict]) -> str:
        """Determine primary market bias"""
        
        bullish_score = 0
        bearish_score = 0
        
        # News-based bias
        if news_data:
            sentiment = news_data.get('avg_sentiment', 0)
            if sentiment > 0.2:
                bullish_score += 30
            elif sentiment > 0.05:
                bullish_score += 15
            elif sentiment < -0.2:
                bearish_score += 30
            elif sentiment < -0.05:
                bearish_score += 15
        
        # Zone-based bias  
        if zone_data:
            current_price = zone_data.get('current_price', 0)
            poc = zone_data.get('poc', 0)
            
            if current_price and poc:
                # Price above POC = bullish bias
                if current_price > poc:
                    bullish_score += 15
                else:
                    bearish_score += 15
                
                # Check position relative to value area
                value_area = zone_data.get('value_area', {})
                va_high = value_area.get('high', 0)
                va_low = value_area.get('low', 0)
                
                if va_high and va_low:
                    if current_price > va_high:
                        bullish_score += 20
                    elif current_price < va_low:
                        bearish_score += 20
        
        # Determine bias
        if bullish_score - bearish_score > 15:
            return "Long"
        elif bearish_score - bullish_score > 15:
            return "Short"
        else:
            return "Neutral"
    
    def check_major_events(self, news_data: Optional[Dict]) -> Dict[str, bool]:
        """Check for major market events"""
        events = {
            'fomc': False,
            'cpi': False,
            'nfp': False,
            'earnings': False
        }
        
        if news_data and 'events' in news_data:
            events.update(news_data['events'])
        
        # Also check date-based events
        now = datetime.utcnow()
        
        # FOMC typically first Wednesday of month (approximation)
        if now.weekday() == 2 and 1 <= now.day <= 7:
            events['fomc'] = True
        
        # CPI typically around 13th of month
        if 10 <= now.day <= 15:
            events['cpi'] = True
        
        # NFP first Friday of month
        if now.weekday() == 4 and 1 <= now.day <= 7:
            events['nfp'] = True
        
        return events
    
    def generate_trade_setups(self, regime: str, bias: str, confidence: float, events: Dict[str, bool]) -> List[Dict[str, Any]]:
        """Generate specific trade setups based on market conditions"""
        setups = []
        
        # Only generate setups if confidence is reasonable
        if confidence < 0.4:
            return setups
        
        # Regime-specific setups
        if regime == "Volatile":
            # Mean reversion setups for volatility
            setups.append({
                "timeWindow": "Opening30Min",
                "direction": "Short" if bias == "Long" else "Long",  # Fade the bias in volatile markets
                "confidenceScore": confidence * 0.8,  # Reduce confidence for volatile setups
                "suggestedRiskMultiple": 0.5,  # Smaller size in volatility
                "rationale": f"Mean reversion setup in {regime} regime, fading {bias} bias"
            })
            
        elif regime == "Trending":
            # Momentum setups for trending
            direction = bias if bias in ["Long", "Short"] else "Long"
            setups.append({
                "timeWindow": "MidMorning",
                "direction": direction,
                "confidenceScore": confidence,
                "suggestedRiskMultiple": 1.5,  # Larger size in trends
                "rationale": f"Momentum setup following {regime} regime with {bias} bias"
            })
            
        elif regime == "Ranging":
            # Range trading setups
            setups.append({
                "timeWindow": "Afternoon",
                "direction": "Short" if bias == "Long" else "Long",  # Fade extremes in ranges
                "confidenceScore": confidence * 0.9,
                "suggestedRiskMultiple": 1.0,
                "rationale": f"Range fade setup in {regime} market"
            })
        
        # Event-specific adjustments
        if events.get('fomc') or events.get('cpi'):
            # Reduce all suggested risk on major event days
            for setup in setups:
                setup['suggestedRiskMultiple'] *= 0.5
                setup['rationale'] += " (reduced size for major event)"
        
        return setups
    
    def save_intelligence(self, intelligence: Dict[str, Any]):
        """Save intelligence to files"""
        signals_dir = os.path.join(self.base_path, "signals")
        os.makedirs(signals_dir, exist_ok=True)
        
        # Save timestamped version
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        timestamped_file = os.path.join(signals_dir, f"signals_{timestamp}.json")
        
        with open(timestamped_file, 'w') as f:
            json.dump(intelligence, f, indent=2)
        
        # Save as latest.json for bot consumption
        latest_file = os.path.join(signals_dir, "latest.json")
        with open(latest_file, 'w') as f:
            json.dump(intelligence, f, indent=2)
        
        logger.info(f"Intelligence saved to {latest_file}")

def main():
    """Main function"""
    generator = EnhancedSignalGenerator()
    intelligence = generator.generate_market_intelligence()
    
    print(f"✅ Enhanced signal generation complete:")
    print(f"   Regime: {intelligence['regime']}")
    print(f"   Confidence: {intelligence['modelConfidence']:.1%}")
    print(f"   Bias: {intelligence['primaryBias']}")
    print(f"   News Intensity: {intelligence['newsIntensity']:.1f}")
    print(f"   Setups: {len(intelligence['setups'])}")
    
    if intelligence['isFomcDay']:
        print("   🚨 FOMC Day detected!")
    if intelligence['isCpiDay']:
        print("   📊 CPI Day detected!")

if __name__ == "__main__":
    main()