#!/usr/bin/env python3
"""
TopstepX Connection Setup for Paper Trading

This script helps you configure your trading bot to connect to TopstepX for actual paper trading.
It will guide you through setting up your credentials and testing the connection.
"""

import os
import sys
import json
import asyncio
import aiohttp
from typing import Optional, Dict, Any

class TopstepXSetup:
    def __init__(self):
        self.api_base = "https://api.topstepx.com"
        self.user_hub = "https://rtc.topstepx.com/hubs/user"
        self.market_hub = "https://rtc.topstepx.com/hubs/market"
        
    def print_banner(self):
        print("""
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                         🔐 TOPSTEPX CONNECTION SETUP 🔐                               ║
║                                                                                       ║
║  This will configure your bot to connect to TopstepX for ACTUAL PAPER TRADING        ║
║                                                                                       ║
║  📋 What you need:                                                                   ║
║  • TopstepX username                                                                 ║
║  • TopstepX API key                                                                  ║
║  • Account ID (optional - will auto-detect)                                         ║
║                                                                                       ║
║  🔒 Security: Credentials stored in local .env file only                            ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
        """)

    def get_credentials(self) -> Dict[str, str]:
        """Interactive credential collection"""
        print("\\n🔐 Enter your TopstepX credentials:")
        print("(These will be stored securely in your .env file)")
        
        username = input("\\n📧 TopstepX Username: ").strip()
        if not username:
            print("❌ Username is required!")
            sys.exit(1)
            
        api_key = input("🔑 TopstepX API Key: ").strip()
        if not api_key:
            print("❌ API key is required!")
            sys.exit(1)
            
        return {
            'username': username,
            'api_key': api_key
        }

    async def test_connection(self, username: str, api_key: str) -> Optional[Dict[str, Any]]:
        """Test TopstepX connection and get JWT token"""
        print(f"\\n🔌 Testing connection to {self.api_base}...")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Login to get JWT token
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
                            
                            # Test JWT validation
                            headers = {'Authorization': f'Bearer {jwt_token}'}
                            async with session.post(
                                f"{self.api_base}/api/Auth/validate",
                                headers=headers
                            ) as validate_response:
                                if validate_response.status == 200:
                                    print("✅ JWT token validation successful!")
                                    return {
                                        'jwt_token': jwt_token,
                                        'username': username,
                                        'api_key': api_key
                                    }
                                else:
                                    print(f"⚠️ JWT validation failed: {validate_response.status}")
                                    return None
                        else:
                            print("❌ No token received in response")
                            return None
                    else:
                        error_text = await response.text()
                        print(f"❌ Authentication failed: {response.status}")
                        print(f"Error: {error_text}")
                        return None
                        
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return None

    def update_env_file(self, credentials: Dict[str, str]):
        """Update .env file with TopstepX credentials"""
        env_path = ".env"
        
        # Read existing .env file
        env_lines = []
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                env_lines = f.readlines()
        
        # Update or add TopstepX credentials
        updated_vars = {
            'TOPSTEPX_USERNAME': credentials['username'],
            'TOPSTEPX_API_KEY': credentials['api_key'],
            'TOPSTEPX_JWT': credentials.get('jwt_token', ''),
            'TOPSTEPX_API_BASE': self.api_base,
            'TOPSTEPX_RTC_BASE': 'https://rtc.topstepx.com',
            'RTC_USER_HUB': self.user_hub,
            'RTC_MARKET_HUB': self.market_hub,
            'DEMO_MODE': '0',  # Disable demo mode
            'PAPER_MODE': '1',  # Enable paper trading mode
            'ENABLE_TOPSTEPX': '1',  # Enable TopstepX connection
            'SKIP_LIVE_CONNECTION': '0'  # Connect to live API
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
        with open(env_path, 'w') as f:
            f.writelines(new_lines)
        
        print(f"✅ Updated {env_path} with TopstepX credentials")

    def print_next_steps(self):
        print("""
🎉 TOPSTEPX CONNECTION CONFIGURED SUCCESSFULLY!

📋 Next Steps:
1. 🛑 Stop the current bot (Ctrl+C in the terminal where it's running)
2. 🚀 Restart the bot to pick up new credentials:
   cd "src\\UnifiedOrchestrator"
   dotnet run

🔍 What to expect:
• Bot will connect to live TopstepX API
• You'll see "✅ Live TopstepX mode enabled" instead of "🎭 Demo mode"
• Paper trades will be executed on TopstepX platform
• All trades will be simulated (no real money at risk)

⚠️ Important Notes:
• This is PAPER TRADING mode - no real money involved
• All trades are simulated for learning purposes
• Monitor the bot logs for connection status
• Check TopstepX dashboard to see paper trade activity

🔧 Troubleshooting:
• If connection fails, check your credentials in .env file
• Ensure TopstepX account has API access enabled
• Check bot logs for detailed error messages
        """)

    async def run_setup(self):
        """Main setup process"""
        self.print_banner()
        
        # Get credentials from user
        credentials = self.get_credentials()
        
        # Test connection
        result = await self.test_connection(credentials['username'], credentials['api_key'])
        
        if result:
            # Update .env file
            self.update_env_file(result)
            self.print_next_steps()
        else:
            print("❌ Setup failed. Please check your credentials and try again.")
            sys.exit(1)

def main():
    """Main entry point"""
    setup = TopstepXSetup()
    
    # Check if we're in the right directory
    if not os.path.exists('src/UnifiedOrchestrator'):
        print("❌ Error: Please run this script from the trading bot root directory")
        print("Expected to find: src/UnifiedOrchestrator")
        sys.exit(1)
    
    # Run async setup
    asyncio.run(setup.run_setup())

if __name__ == "__main__":
    main()
