#!/usr/bin/env python3
"""
Fetch 90 days of historical data from TopstepX and save it for bot practice.
This pre-loads the data so the bot can practice immediately without waiting for SDK initialization.

Supports two modes:
1. FULL refresh: Fetch 90 days, replace entire file
2. INCREMENTAL: Fetch only new bars since last update, append and trim to 90 days
"""

import os
import sys
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path to import TopstepXAdapter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'adapters'))

from topstep_x_adapter import TopstepXAdapter

# Configuration
REFRESH_MODE = os.getenv('REFRESH_MODE', 'incremental')  # 'full' or 'incremental'
LOOKBACK_DAYS = 90  # Keep last 90 days


def load_existing_data(file_path):
    """Load existing historical data from file."""
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"   ⚠️ Could not load existing file: {e}")
        return None


def get_last_timestamp(existing_data):
    """Get the timestamp of the last bar in existing data."""
    if not existing_data or not existing_data.get('bars'):
        return None
    
    bars = existing_data['bars']
    if not bars:
        return None
    
    last_bar = bars[-1]
    timestamp_str = last_bar.get('timestamp')
    
    if not timestamp_str:
        return None
    
    try:
        # Parse timestamp (format: "2025-08-31 17:00:00-05:00")
        # Remove timezone for datetime parsing
        ts_without_tz = timestamp_str.rsplit('-', 1)[0].rsplit('+', 1)[0].strip()
        return datetime.strptime(ts_without_tz, '%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"   ⚠️ Could not parse timestamp: {e}")
        return None


def trim_to_lookback_days(bars, lookback_days):
    """Keep only the most recent bars within lookback window."""
    if not bars:
        return bars
    
    # Find cutoff date
    cutoff = datetime.now() - timedelta(days=lookback_days)
    
    # Filter bars
    trimmed = []
    for bar in bars:
        timestamp_str = bar.get('timestamp', '')
        try:
            ts_without_tz = timestamp_str.rsplit('-', 1)[0].rsplit('+', 1)[0].strip()
            bar_time = datetime.strptime(ts_without_tz, '%Y-%m-%d %H:%M:%S')
            
            if bar_time >= cutoff:
                trimmed.append(bar)
        except:
            # Keep bar if we can't parse (safer)
            trimmed.append(bar)
    
    return trimmed


async def fetch_and_save_historical_data():
    """Fetch 90 days of historical data and save to disk for bot practice."""
    
    # Check environment variables
    api_key = os.getenv('TOPSTEPX_API_KEY')
    username = os.getenv('TOPSTEPX_USERNAME')
    
    if not api_key or not username:
        print("❌ ERROR: TOPSTEPX_API_KEY and TOPSTEPX_USERNAME must be set in .env")
        return False
    
    print(f"✅ Environment variables loaded")
    print(f"   Username: {username}")
    print(f"   API Key: {'*' * 20}{api_key[-4:]}")
    print(f"   Refresh Mode: {REFRESH_MODE.upper()}")
    print()
    
    # Symbols to fetch
    symbols = ['ES', 'NQ']
    
    # Create data directory
    data_dir = Path('data/historical')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for symbol in symbols:
        print(f"🔍 Fetching {symbol} data...")
        
        # Check for existing data
        file_path = data_dir / f'{symbol}_90days.json'
        existing_data = load_existing_data(file_path)
        
        # Determine date range based on mode
        end_date = datetime.now()
        
        if REFRESH_MODE == 'incremental' and existing_data:
            # Incremental: Fetch only new bars since last update
            last_timestamp = get_last_timestamp(existing_data)
            
            if last_timestamp:
                start_date = last_timestamp + timedelta(minutes=5)  # Start after last bar
                print(f"   📊 Incremental update from last bar: {last_timestamp.strftime('%Y-%m-%d %H:%M')}")
                
                # If last bar is very recent (< 10 minutes), nothing to fetch
                time_since_last = (end_date - last_timestamp).total_seconds() / 60
                if time_since_last < 10:
                    print(f"   ✅ Data is up-to-date (last bar {int(time_since_last)} min ago)")
                    continue
            else:
                # No valid timestamp, fall back to full refresh
                print(f"   ⚠️ Could not find last timestamp, doing full refresh")
                start_date = end_date - timedelta(days=LOOKBACK_DAYS)
        else:
            # Full refresh: Fetch entire lookback window
            start_date = end_date - timedelta(days=LOOKBACK_DAYS)
            print(f"   🔄 Full refresh mode")
        
        print(f"   � Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print()
        
        try:
            # Create adapter
            adapter = TopstepXAdapter(instruments=[symbol])
            
            # Initialize with stdout redirect
            print(f"   Initializing TopstepX SDK...")
            original_stdout = sys.stdout
            sys.stdout = sys.stderr
            
            try:
                await asyncio.wait_for(adapter.initialize(), timeout=60.0)
            finally:
                sys.stdout = original_stdout
            
            print(f"   ✅ TopstepX SDK initialized")
            
            # Fetch in chunks to get full 90 days
            # API limit: 1000 bars/request
            # 5min bars: 1000 bars = ~13 trading days
            # Need ~7 chunks for 90 calendar days
            print(f"   Fetching bars in chunks (API limit: 1000 bars/request)...")
            
            all_bars = []
            chunk_start = start_date
            chunk_number = 1
            
            while chunk_start < end_date:
                # Each chunk: ~15 calendar days to ensure we get 1000 bars
                chunk_end = min(chunk_start + timedelta(days=15), end_date)
                
                print(f"   📦 Chunk {chunk_number}: {chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}...")
                
                bars_result = await adapter.fetch_historical_bars(
                    symbol=symbol,
                    start_date=chunk_start,
                    end_date=chunk_end
                )
                
                if bars_result and bars_result.get('success'):
                    # Extract bars from result
                    bars_data = bars_result.get('bars', {})
                    if isinstance(bars_data, dict):
                        chunk_bars = bars_data.get('bars', [])
                    else:
                        chunk_bars = bars_data
                    
                    if chunk_bars:
                        print(f"      ✅ Fetched {len(chunk_bars)} bars")
                        all_bars.extend(chunk_bars)
                    else:
                        print(f"      ⚠️ No data returned")
                else:
                    print(f"      ⚠️ Request failed: {bars_result.get('error', 'Unknown error')}")
                
                # Move to next chunk
                chunk_start = chunk_end
                chunk_number += 1
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)
            
            bar_count = len(all_bars)
            
            if bar_count == 0:
                if REFRESH_MODE == 'incremental' and existing_data:
                    print(f"   ℹ️ No new bars to fetch (data already up-to-date)")
                    continue
                else:
                    print(f"   ❌ No data fetched for {symbol}")
                    continue
            
            print(f"   ✅ Fetched: {bar_count} new bars")
            
            # Merge with existing data if incremental mode
            if REFRESH_MODE == 'incremental' and existing_data:
                existing_bars = existing_data.get('bars', [])
                print(f"   🔗 Merging with {len(existing_bars)} existing bars...")
                
                # Combine: existing + new
                combined_bars = existing_bars + all_bars
                
                # Sort by timestamp
                combined_bars.sort(key=lambda b: b.get('timestamp', ''))
                
                # Remove duplicates
                seen_timestamps = set()
                bars_list = []
                for bar in combined_bars:
                    ts = bar.get('timestamp')
                    if ts and ts not in seen_timestamps:
                        seen_timestamps.add(ts)
                        bars_list.append(bar)
                
                # Trim to lookback window (keep last 90 days)
                bars_list = trim_to_lookback_days(bars_list, LOOKBACK_DAYS)
                
                bar_count = len(bars_list)
                print(f"   ✅ After merge and trim: {bar_count} bars (last {LOOKBACK_DAYS} days)")
            else:
                # Full refresh mode: just use fetched bars
                # Sort by timestamp and remove duplicates
                all_bars.sort(key=lambda b: b.get('timestamp', ''))
            
            seen_timestamps = set()
            bars_list = []
            for bar in all_bars:
                ts = bar.get('timestamp')
                if ts and ts not in seen_timestamps:
                    seen_timestamps.add(ts)
                    bars_list.append(bar)
            
            if len(bars_list) < bar_count:
                print(f"   🔧 Removed {bar_count - len(bars_list)} duplicate bars")
                bar_count = len(bars_list)
            
            if bar_count == 0:
                print(f"   ⚠️  No data returned for {symbol}")
                continue
            
            # Save to file
            output_file = data_dir / f'{symbol}_90days.json'
            with open(output_file, 'w') as f:
                json.dump({
                    'symbol': symbol,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'bar_count': bar_count,
                    'bars': bars_list,
                    'fetched_at': datetime.now().isoformat()
                }, f, indent=2)
            
            print(f"   💾 Saved to: {output_file}")
            
            # Show sample bars
            if bar_count >= 5:
                print(f"\n   📊 Sample bars (first 3):")
                for i, bar in enumerate(bars_list[:3], 1):
                    timestamp = bar.get('timestamp', 'N/A')
                    open_price = bar.get('open', 0)
                    high = bar.get('high', 0)
                    low = bar.get('low', 0)
                    close = bar.get('close', 0)
                    volume = bar.get('volume', 0)
                    print(f"      [{i}] {timestamp}")
                    print(f"          O: {open_price}, H: {high}, L: {low}, C: {close}, V: {volume}")
                
                print(f"\n   📊 Sample bars (last 3):")
                for i, bar in enumerate(bars_list[-3:], bar_count - 2):
                    timestamp = bar.get('timestamp', 'N/A')
                    open_price = bar.get('open', 0)
                    high = bar.get('high', 0)
                    low = bar.get('low', 0)
                    close = bar.get('close', 0)
                    volume = bar.get('volume', 0)
                    print(f"      [{i}] {timestamp}")
                    print(f"          O: {open_price}, H: {high}, L: {low}, C: {close}, V: {volume}")
            
            print()
            
        except Exception as e:
            print(f"   ❌ Error fetching {symbol} data: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("=" * 60)
    print("✅ HISTORICAL DATA FETCH COMPLETED")
    print(f"📁 Data saved to: {data_dir.absolute()}")
    print()
    print("🤖 Bot can now practice using:")
    print("   - data/historical/ES_90days.json")
    print("   - data/historical/NQ_90days.json")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    # Run the async function
    success = asyncio.run(fetch_and_save_historical_data())
    sys.exit(0 if success else 1)
