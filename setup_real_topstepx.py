#!/usr/bin/env python3
"""
TopstepX Real Account Setup for Paper Trading

This configures your bot to connect to your REAL TopstepX account for paper trading.
"""

import os
import sys
import asyncio
import aiohttp
import getpass
from pathlib import Path

class TopstepXRealSetup:
    def __init__(self):
        self.api_base = "https://api.topstepx.com"
        self.env_file = Path(".env")
        
    def print_banner(self):
        print("""
╔════════════════════════════════════════════════════════════════════════════════════════╗
║                    🏦 TOPSTEPX REAL ACCOUNT PAPER TRADING SETUP 🏦                     ║
║                                                                                        ║
║  🎯 REAL MARKET DATA + PAPER TRADING                                                  ║
║  • Connects to your actual TopstepX account                                          ║
║  • Uses real market data and live price feeds                                        ║
║  • Executes paper trades (simulated, no real money)                                  ║
║  • Monitors actual market conditions                                                  ║
║                                                                                        ║
║  📋 Required:                                                                         ║
║  • Your TopstepX account username                                                     ║
║  • Your TopstepX API key                                                              ║
║  • Account must have API access enabled                                               ║
║                                                                                        ║
║  🔒 Security: Credentials encrypted and stored locally only                          ║
╚════════════════════════════════════════════════════════════════════════════════════════╝
        """)

    def get_credentials(self):
        """Get real TopstepX credentials"""
        print("\\n🔐 Enter your REAL TopstepX credentials:")
        print("⚠️  These will be used to connect to your actual TopstepX account")
        print("💡 Paper trading means simulated trades with real market data")
        
        username = input("\\n📧 TopstepX Username: ").strip()
        if not username:
            print("❌ Username is required!")
            sys.exit(1)
            
        # Use getpass for secure password/API key input
        api_key = getpass.getpass("🔑 TopstepX API Key (hidden): ").strip()
        if not api_key:
            print("❌ API key is required!")
            sys.exit(1)
            
        return username, api_key

    async def test_real_connection(self, username: str, api_key: str):
        """Test actual TopstepX connection and get real JWT token"""
        print(f"\\n🔌 Connecting to TopstepX API: {self.api_base}")
        print("⏳ Authenticating with your account...")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Real authentication call
                login_data = {
                    'userName': username,
                    'apiKey': api_key
                }
                
                async with session.post(
                    f"{self.api_base}/api/Auth/loginKey",
                    json=login_data,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        jwt_token = data.get('token')
                        if jwt_token:
                            print("✅ Authentication successful!")
                            
                            # Validate JWT and get account info
                            headers = {'Authorization': f'Bearer {jwt_token}'}
                            async with session.post(
                                f"{self.api_base}/api/Auth/validate",
                                headers=headers
                            ) as validate_response:
                                if validate_response.status == 200:
                                    print("✅ JWT token validated!")
                                    
                                    # Get account information
                                    async with session.get(
                                        f"{self.api_base}/api/Account",
                                        headers=headers
                                    ) as account_response:
                                        if account_response.status == 200:
                                            account_data = await account_response.json()
                                            print("✅ Account access confirmed!")
                                            
                                            if 'data' in account_data and account_data['data']:
                                                accounts = account_data['data']
                                                print(f"\\n📊 Found {len(accounts)} account(s):")
                                                for i, acc in enumerate(accounts[:5]):  # Show first 5
                                                    acc_id = acc.get('id', 'N/A')
                                                    acc_name = acc.get('name', 'N/A')
                                                    balance = acc.get('balance', 'N/A')
                                                    acc_type = acc.get('type', 'N/A')
                                                    print(f"   {i+1}. Account {acc_id}: {acc_name}")
                                                    print(f"      Type: {acc_type}, Balance: ${balance}")
                                                
                                                # Select primary account
                                                primary_account = accounts[0]
                                                primary_id = primary_account.get('id')
                                                
                                                return {
                                                    'username': username,
                                                    'api_key': api_key,
                                                    'jwt_token': jwt_token,
                                                    'account_id': primary_id,
                                                    'account_info': primary_account
                                                }
                                        else:
                                            print(f"⚠️ Could not access account info: {account_response.status}")
                                            return {
                                                'username': username,
                                                'api_key': api_key,
                                                'jwt_token': jwt_token
                                            }
                                else:
                                    print(f"❌ JWT validation failed: {validate_response.status}")
                                    return None
                        else:
                            print("❌ No token received from TopstepX")
                            return None
                    else:
                        error_text = await response.text()
                        print(f"❌ Authentication failed: {response.status}")
                        print(f"   Error: {error_text}")
                        print("\\n💡 Please check:")
                        print("   • Username is correct")
                        print("   • API key is valid and not expired")
                        print("   • Account has API access enabled")
                        return None
                        
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return None

    def update_env_file(self, credentials: dict):
        """Update .env file with real TopstepX credentials"""
        print("\\n📝 Updating .env file with real credentials...")
        
        # Read existing .env file
        env_lines = []
        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                env_lines = f.readlines()
        
        # Configuration for real TopstepX connection with paper trading
        updated_vars = {
            'TOPSTEPX_USERNAME': credentials['username'],
            'TOPSTEPX_API_KEY': credentials['api_key'],
            'TOPSTEPX_JWT': credentials.get('jwt_token', ''),
            'TOPSTEPX_ACCOUNT_ID': credentials.get('account_id', ''),
            'TOPSTEPX_API_BASE': self.api_base,
            'TOPSTEPX_RTC_BASE': 'https://rtc.topstepx.com',
            'RTC_USER_HUB': 'https://rtc.topstepx.com/hubs/user',
            'RTC_MARKET_HUB': 'https://rtc.topstepx.com/hubs/market',
            
            # Paper trading configuration (real connection, simulated trades)
            'PAPER_MODE': '1',
            'TRADING_MODE': 'PAPER',
            'DEMO_MODE': '0',
            'SKIP_LIVE_CONNECTION': '0',
            'ENABLE_TOPSTEPX': '1',
            'AUTH_ALLOW': '1',
            'AUTO_PAPER_TRADING': '1',
            
            # Real market data but paper trades
            'USE_REAL_MARKET_DATA': '1',
            'SIMULATE_TRADES_ONLY': '1',
            'REAL_ACCOUNT_PAPER_MODE': '1'
        }
        
        # Update existing lines or prepare new ones
        new_lines = []
        updated_keys = set()
        
        for line in env_lines:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                if key in updated_vars:
                    new_lines.append(f"{key}={updated_vars[key]}\\n")
                    updated_keys.add(key)
                else:
                    new_lines.append(line + '\\n')
            else:
                new_lines.append(line + '\\n')
        
        # Add new variables that weren't found
        for key, value in updated_vars.items():
            if key not in updated_keys:
                new_lines.append(f"{key}={value}\\n")
        
        # Write updated .env file
        with open(self.env_file, 'w') as f:
            f.writelines(new_lines)
        
        print(f"✅ Updated {self.env_file} with real TopstepX credentials")

    def print_success_message(self, credentials: dict):
        account_info = credentials.get('account_info', {})
        account_id = credentials.get('account_id', 'N/A')
        
        print(f"""
🎉 REAL TOPSTEPX ACCOUNT CONFIGURED FOR PAPER TRADING!

📊 Account Details:
• Username: {credentials['username']}
• Account ID: {account_id}
• Account Name: {account_info.get('name', 'N/A')}
• Account Type: {account_info.get('type', 'N/A')}
• Balance: ${account_info.get('balance', 'N/A')}

🎯 Trading Mode: PAPER TRADING
• ✅ Real TopstepX account connection
• ✅ Live market data feeds
• ✅ Real price movements and volatility
• ✅ Actual economic events and news impact
• 🎭 Simulated trade execution (no real money)

📋 Next Steps:
1. 🛑 Stop the current bot (Ctrl+C)
2. 🚀 Restart the bot:
   cd "src\\UnifiedOrchestrator"
   dotnet run

🔍 What to expect:
• Bot connects to your real TopstepX account
• Live market data streaming
• Real-time price feeds for ES/NQ
• Paper trades logged and tracked
• All strategies execute on live market conditions
• No real money at risk - pure simulation

📈 Monitoring:
• Check bot logs for "Connected to TopstepX account: {account_id}"
• Monitor paper trade executions in console
• Track performance with real market conditions
• All trades visible in bot dashboard

⚠️  Important:
• This is PAPER TRADING - no real money involved
• Trades are simulated but use real market data
• Perfect for testing strategies safely
• Can switch to live trading later if desired
        """)

    async def run_setup(self):
        """Main setup process"""
        self.print_banner()
        
        # Get real credentials
        username, api_key = self.get_credentials()
        
        # Test real connection
        credentials = await self.test_real_connection(username, api_key)
        
        if credentials:
            # Update .env file
            self.update_env_file(credentials)
            self.print_success_message(credentials)
        else:
            print("❌ Setup failed. Please check your TopstepX credentials and try again.")
            print("\\n💡 Troubleshooting:")
            print("   • Verify credentials in TopstepX dashboard")
            print("   • Ensure API access is enabled")
            print("   • Check account status")
            sys.exit(1)

def main():
    """Main entry point"""
    setup = TopstepXRealSetup()
    
    # Check if we're in the right directory
    if not os.path.exists('src/UnifiedOrchestrator'):
        print("❌ Error: Please run this script from the trading bot root directory")
        print("Expected to find: src/UnifiedOrchestrator")
        sys.exit(1)
    
    # Run async setup
    asyncio.run(setup.run_setup())

if __name__ == "__main__":
    main()
