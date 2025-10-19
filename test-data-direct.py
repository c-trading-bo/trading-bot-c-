"""
Direct test of TopstepX data retrieval (live + historical)
Tests authentication and data access without starting the bot
"""
import asyncio
import os
from dotenv import load_dotenv
from project_x_py import TradingSuite

async def test_data_access():
    # Load environment variables
    load_dotenv()
    
    print("=" * 60)
    print("TopstepX Data Access Test")
    print("=" * 60)
    
    # Check environment
    api_key = os.getenv('PROJECT_X_API_KEY')
    username = os.getenv('PROJECT_X_USERNAME')
    
    print(f"\n✓ API Key present: {api_key is not None}")
    print(f"✓ Username: {username}")
    
    try:
        print("\n[1/3] Authenticating with TopstepX...")
        suite = await TradingSuite.from_env(['ES', 'NQ'])
        print("✅ Authentication successful!")
        
        print("\n[2/3] Testing LIVE data...")
        es_price = await suite["ES"].data.get_current_price()
        nq_price = await suite["NQ"].data.get_current_price()
        
        print(f"✅ ES Live Price: ${es_price:,.2f}")
        print(f"✅ NQ Live Price: ${nq_price:,.2f}")
        
        print("\n[3/3] Testing HISTORICAL data...")
        es_bars = await suite["ES"].data.get_latest_bars(count=10)
        nq_bars = await suite["NQ"].data.get_latest_bars(count=10)
        
        if not es_bars.is_empty():
            es_data = es_bars.to_dicts()
            print(f"✅ ES Historical: Retrieved {len(es_data)} bars")
            print(f"   Latest bar: {es_data[-1]['timestamp']} - Close: ${es_data[-1]['close']:,.2f}")
        else:
            print("⚠️  ES Historical: No data returned")
        
        if not nq_bars.is_empty():
            nq_data = nq_bars.to_dicts()
            print(f"✅ NQ Historical: Retrieved {len(nq_data)} bars")
            print(f"   Latest bar: {nq_data[-1]['timestamp']} - Close: ${nq_data[-1]['close']:,.2f}")
        else:
            print("⚠️  NQ Historical: No data returned")
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - TopstepX data access working!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED")
        print("=" * 60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        
        # Check for authentication errors
        if "Authentication failed" in str(e) or "401" in str(e):
            print("\n🔍 DIAGNOSIS: API Key Authentication Problem")
            print("   → Check TopstepX portal: Is your API key ACTIVE?")
            print("   → Verify key has Trading + Market Data permissions")
            print("   → Confirm account has API access enabled")
        
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_data_access())
    exit(0 if success else 1)
