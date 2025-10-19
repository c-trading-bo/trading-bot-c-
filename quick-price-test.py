"""
Quick test to get live ES/NQ prices - recreating the working test from earlier
"""
import asyncio
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

async def get_live_prices():
    print("Loading TopstepX SDK...")
    from project_x_py import TradingSuite
    
    print(f"API Key: {os.getenv('PROJECT_X_API_KEY')[:10]}...")
    print(f"Username: {os.getenv('PROJECT_X_USERNAME')}")
    
    print("\nAuthenticating...")
    suite = await TradingSuite.from_env(['ES', 'NQ'])
    
    print("\nGetting live prices...")
    es_price = await suite["ES"].data.get_current_price()
    nq_price = await suite["NQ"].data.get_current_price()
    
    print(f"\n✅ ES Live Price: ${es_price:,.2f}")
    print(f"✅ NQ Live Price: ${nq_price:,.2f}")
    
    return es_price, nq_price

if __name__ == "__main__":
    try:
        es, nq = asyncio.run(get_live_prices())
        print(f"\n🎉 SUCCESS! Got live market data.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
