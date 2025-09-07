#!/usr/bin/env python3
"""
Quick TopstepX Connection Test

Tests your TopstepX connection without modifying any files.
"""

import os
import sys
import asyncio
import aiohttp
from dotenv import load_dotenv

async def test_topstepx_connection():
    """Test TopstepX connection using current environment variables"""
    
    # Load environment variables
    load_dotenv()
    
    username = os.getenv('TOPSTEPX_USERNAME')
    api_key = os.getenv('TOPSTEPX_API_KEY')
    jwt_token = os.getenv('TOPSTEPX_JWT')
    api_base = os.getenv('TOPSTEPX_API_BASE', 'https://api.topstepx.com')
    
    print(f"""
🔍 TOPSTEPX CONNECTION TEST
═══════════════════════════

📋 Configuration:
• API Base: {api_base}
• Username: {'✅ Set' if username else '❌ Not set'}
• API Key: {'✅ Set' if api_key else '❌ Not set'}  
• JWT Token: {'✅ Set' if jwt_token else '❌ Not set'}
    """)
    
    if not username or not api_key:
        print("❌ Missing required credentials!")
        print("   Run: python setup_topstepx_connection.py")
        return False
    
    try:
        async with aiohttp.ClientSession() as session:
            print("🔌 Testing authentication...")
            
            # Test login
            login_data = {
                'userName': username,
                'apiKey': api_key
            }
            
            async with session.post(
                f"{api_base}/api/Auth/loginKey",
                json=login_data,
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    new_jwt = data.get('token')
                    if new_jwt:
                        print("✅ Authentication successful!")
                        
                        # Test JWT validation
                        headers = {'Authorization': f'Bearer {new_jwt}'}
                        async with session.post(
                            f"{api_base}/api/Auth/validate",
                            headers=headers
                        ) as validate_response:
                            if validate_response.status == 200:
                                print("✅ JWT validation successful!")
                                
                                # Test account access (optional)
                                async with session.get(
                                    f"{api_base}/api/Account",
                                    headers=headers
                                ) as account_response:
                                    if account_response.status == 200:
                                        account_data = await account_response.json()
                                        print(f"✅ Account access confirmed!")
                                        if 'data' in account_data and account_data['data']:
                                            accounts = account_data['data']
                                            print(f"📊 Found {len(accounts)} account(s)")
                                            for acc in accounts[:3]:  # Show first 3
                                                acc_id = acc.get('id', 'N/A')
                                                acc_name = acc.get('name', 'N/A')
                                                balance = acc.get('balance', 'N/A')
                                                print(f"   • Account {acc_id}: {acc_name} (Balance: ${balance})")
                                    else:
                                        print(f"⚠️ Account access failed: {account_response.status}")
                                
                                print("""
🎉 CONNECTION TEST SUCCESSFUL!

Your bot is ready to connect to TopstepX for paper trading.
Restart your bot to see it connect to live TopstepX API.
                                """)
                                return True
                            else:
                                print(f"❌ JWT validation failed: {validate_response.status}")
                                return False
                    else:
                        print("❌ No token received")
                        return False
                else:
                    error_text = await response.text()
                    print(f"❌ Authentication failed: {response.status}")
                    print(f"   Error: {error_text}")
                    return False
                    
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def main():
    """Main entry point"""
    print("🔍 Testing TopstepX connection...")
    
    # Check if aiohttp is available
    try:
        import aiohttp
    except ImportError:
        print("❌ aiohttp not installed. Installing...")
        os.system("pip install aiohttp python-dotenv")
        import aiohttp
    
    # Run test
    success = asyncio.run(test_topstepx_connection())
    
    if not success:
        print("\\n❌ Connection test failed.")
        print("   Run: python setup_topstepx_connection.py")
        sys.exit(1)

if __name__ == "__main__":
    main()
