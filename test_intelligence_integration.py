#!/usr/bin/env python3
"""
Integration test for the intelligence pipeline
"""

import sys
import os
import json
from datetime import datetime

# Add Intelligence scripts to path
sys.path.append('Intelligence/scripts')

def test_intelligence_pipeline():
    """Test the complete intelligence pipeline"""
    
    print("🧪 Testing Intelligence Pipeline Integration")
    print("=" * 50)
    
    # Test 1: News Collection
    print("\n1. Testing News Collection...")
    try:
        from collect_news_multi import NewsCollector
        collector = NewsCollector()
        
        # Run a quick test with timeout
        news_results = collector.collect_all_news()
        
        print(f"   ✅ News collection completed")
        print(f"   📰 Articles collected: {news_results['article_count']}")
        print(f"   😊 Average sentiment: {news_results['avg_sentiment']:.2f}")
        print(f"   📊 News intensity: {news_results['news_intensity']:.1f}")
        print(f"   🏛️ Regime hint: {news_results['regime_hint']}")
        
    except Exception as e:
        print(f"   ❌ News collection failed: {e}")
        return False
    
    # Test 2: Signal Generation
    print("\n2. Testing Signal Generation...")
    try:
        from generate_signals_enhanced import EnhancedSignalGenerator
        generator = EnhancedSignalGenerator()
        
        intelligence = generator.generate_market_intelligence()
        
        print(f"   ✅ Signal generation completed")
        print(f"   🎯 Market regime: {intelligence['regime']}")
        print(f"   📊 Model confidence: {intelligence['modelConfidence']:.1%}")
        print(f"   📈 Primary bias: {intelligence['primaryBias']}")
        print(f"   📰 News intensity: {intelligence['newsIntensity']:.1f}")
        print(f"   🎬 Trade setups: {len(intelligence['setups'])}")
        
        if intelligence['isFomcDay']:
            print("   🚨 FOMC Day detected!")
        if intelligence['isCpiDay']:
            print("   📊 CPI Day detected!")
            
    except Exception as e:
        print(f"   ❌ Signal generation failed: {e}")
        return False
    
    # Test 3: File Output Verification
    print("\n3. Testing File Output...")
    
    signals_file = "Intelligence/data/signals/latest.json"
    news_file = "Intelligence/data/news/latest.json"
    zones_file = "Intelligence/data/zones/latest_zones.json"
    
    files_ok = 0
    
    if os.path.exists(signals_file):
        with open(signals_file, 'r') as f:
            signals_data = json.load(f)
            print(f"   ✅ Signals file exists: {signals_data['regime']} regime")
            files_ok += 1
    else:
        print(f"   ❌ Signals file missing: {signals_file}")
    
    if os.path.exists(news_file):
        with open(news_file, 'r') as f:
            news_data = json.load(f)
            print(f"   ✅ News file exists: {news_data['article_count']} articles")
            files_ok += 1
    else:
        print(f"   ❌ News file missing: {news_file}")
    
    if os.path.exists(zones_file):
        with open(zones_file, 'r') as f:
            zones_data = json.load(f)
            supply_count = len(zones_data.get('supply_zones', []))
            demand_count = len(zones_data.get('demand_zones', []))
            print(f"   ✅ Zones file exists: {supply_count} supply, {demand_count} demand zones")
            files_ok += 1
    else:
        print(f"   ❌ Zones file missing: {zones_file}")
    
    # Test 4: C# Model Compatibility
    print("\n4. Testing C# Model Compatibility...")
    
    try:
        # Test signals JSON structure matches C# MarketContext
        if os.path.exists(signals_file):
            with open(signals_file, 'r') as f:
                signals = json.load(f)
                
            required_fields = [
                'date', 'regime', 'newsIntensity', 'isCpiDay', 'isFomcDay',
                'modelConfidence', 'primaryBias', 'setups', 'generatedAt'
            ]
            
            missing_fields = [field for field in required_fields if field not in signals]
            
            if not missing_fields:
                print("   ✅ Signals JSON format matches C# MarketContext model")
            else:
                print(f"   ❌ Missing fields in signals: {missing_fields}")
                return False
        
        # Test zones JSON structure matches C# ZoneData
        if os.path.exists(zones_file):
            with open(zones_file, 'r') as f:
                zones = json.load(f)
                
            required_zone_fields = [
                'symbol', 'current_price', 'supply_zones', 'demand_zones', 
                'poc', 'key_levels', 'generated_at'
            ]
            
            missing_zone_fields = [field for field in required_zone_fields if field not in zones]
            
            if not missing_zone_fields:
                print("   ✅ Zones JSON format matches C# ZoneData model")
            else:
                print(f"   ❌ Missing fields in zones: {missing_zone_fields}")
                return False
            
    except Exception as e:
        print(f"   ❌ C# compatibility test failed: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 50)
    if files_ok == 3:
        print("🎉 Intelligence Pipeline Integration Test: PASSED")
        print(f"   ✅ All {files_ok}/3 output files generated successfully")
        print("   ✅ JSON formats match C# models")
        print("   ✅ News collection and signal generation working")
        print("\n🤖 The C# bot should now be able to consume intelligence data!")
        return True
    else:
        print("❌ Intelligence Pipeline Integration Test: FAILED")
        print(f"   Only {files_ok}/3 output files generated")
        return False

if __name__ == "__main__":
    success = test_intelligence_pipeline()
    sys.exit(0 if success else 1)