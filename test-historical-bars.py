#!/usr/bin/env python3
"""
Test script to verify TopstepX historical data fetching works correctly.
Tests loading 7 days of bars for ES futures before launching full bot.
"""

import asyncio
import json
import sys
import os
from datetime import datetime, timedelta

# Add src/adapters to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'adapters'))

from topstep_x_adapter import TopstepXAdapter

async def test_historical_bars():
    """Test fetching 7 days of historical bars for ES"""
    print("=" * 80)
    print("TOPSTEPX HISTORICAL DATA TEST")
    print("=" * 80)
    
    # Check environment variables
    api_key = os.getenv('TOPSTEPX_API_KEY')
    username = os.getenv('TOPSTEPX_USERNAME')
    
    if not api_key:
        print("❌ TOPSTEPX_API_KEY not found in environment")
        return False
    
    if not username:
        print("❌ TOPSTEPX_USERNAME not found in environment")
        return False
    
    print(f"✅ API Key: {api_key[:10]}...")
    print(f"✅ Username: {username}")
    print()
    
    # Create adapter
    print("📦 Creating TopstepX adapter...")
    adapter = TopstepXAdapter(instruments=['ES'])
    
    # Initialize adapter
    print("🔌 Initializing TopstepX connection...")
    try:
        # Redirect stdout during initialization to prevent SDK prints
        original_stdout = sys.stdout
        sys.stdout = sys.stderr
        
        await asyncio.wait_for(adapter.initialize(), timeout=60.0)
        
        sys.stdout = original_stdout
        print("✅ TopstepX connection established")
    except asyncio.TimeoutError:
        sys.stdout = original_stdout
        print("❌ Connection timed out after 60 seconds")
        return False
    except Exception as e:
        sys.stdout = original_stdout
        print(f"❌ Connection failed: {e}")
        return False
    
    print()
    
    # Calculate date range (7 days ago to now)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)
    
    print(f"📅 Fetching historical bars:")
    print(f"   Symbol: ES")
    print(f"   Start: {start_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"   End: {end_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"   Duration: 7 days")
    print()
    
    # Fetch historical bars
    print("⏳ Fetching bars from TopstepX...")
    try:
        bars_result = await adapter.fetch_historical_bars(
            symbol='ES',
            start_date=start_date,
            end_date=end_date
        )
        
        # Handle dict response (bars might be nested in 'bars' key)
        if isinstance(bars_result, dict):
            bars = bars_result.get('bars', [])
            if not bars and 'data' in bars_result:
                bars = bars_result.get('data', [])
        else:
            bars = bars_result
        
        if not bars:
            print(f"❌ No bars returned (got type: {type(bars_result)})")
            print(f"   Response keys: {bars_result.keys() if isinstance(bars_result, dict) else 'N/A'}")
            return False
        
        print(f"✅ Successfully fetched {len(bars)} bars")
        print()
        
        # Convert to list if needed
        if hasattr(bars, 'values'):
            bars = list(bars.values())
        
        # Show sample data
        print("📊 Sample bars (first 5):")
        for i, bar in enumerate(list(bars)[:5]):
            timestamp = bar.get('timestamp', 'N/A')
            open_price = bar.get('open', 0)
            high_price = bar.get('high', 0)
            low_price = bar.get('low', 0)
            close_price = bar.get('close', 0)
            volume = bar.get('volume', 0)
            
            print(f"   [{i+1}] {timestamp}")
            print(f"       O: {open_price:.2f}, H: {high_price:.2f}, L: {low_price:.2f}, C: {close_price:.2f}, V: {volume}")
        
        print()
        
        # Show last 5 bars
        if len(bars) > 5:
            print("📊 Sample bars (last 5):")
            bars_list = list(bars)
            for i, bar in enumerate(bars_list[-5:]):
                timestamp = bar.get('timestamp', 'N/A')
                open_price = bar.get('open', 0)
                high_price = bar.get('high', 0)
                low_price = bar.get('low', 0)
                close_price = bar.get('close', 0)
                volume = bar.get('volume', 0)
                
                print(f"   [{len(bars_list)-4+i}] {timestamp}")
                print(f"       O: {open_price:.2f}, H: {high_price:.2f}, L: {low_price:.2f}, C: {close_price:.2f}, V: {volume}")
        
        print()
        print("=" * 80)
        print("✅ HISTORICAL DATA TEST PASSED")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to fetch bars: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    success = await test_historical_bars()
    
    if success:
        print("\n✅ Historical data fetching works correctly")
        print("   The bot can now load 7+ days of data for backtesting and learning")
        sys.exit(0)
    else:
        print("\n❌ Historical data fetching failed")
        print("   Please fix the issues before launching the bot")
        sys.exit(1)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
