#!/usr/bin/env python3
"""
TopstepX Connection and Authentication Test
Tests the critical authentication components for live trading
Uses the correct TopstepX API endpoint: /api/Auth/loginKey
"""
import os
import json
import requests
from pathlib import Path

class TopstepXConnectionTest:
    def __init__(self):
        self.load_environment()
        
    def load_environment(self):
        """Load environment variables from .env file"""
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
        
        # Get credentials
        self.api_base = os.getenv('TOPSTEPX_API_BASE', 'https://api.topstepx.com').rstrip('/')
        self.api_key = os.getenv('TOPSTEPX_API_KEY')
        self.username = os.getenv('TOPSTEPX_USERNAME')
        self.account_id = os.getenv('TOPSTEPX_ACCOUNT_ID')

    def print_banner(self):
        print("=" * 60)
        print("🔐 TOPSTEPX CONNECTION & AUTHENTICATION TEST")
        print("=" * 60)

    def check_credentials(self):
        """Check if required credentials are configured"""
        print("\n📋 CREDENTIAL CONFIGURATION CHECK:")
        
        missing_creds = []
        
        print(f"   📍 TOPSTEPX_API_BASE - {self.api_base}")
        
        if not self.api_key:
            print("   ❌ TOPSTEPX_API_KEY - Not configured")
            missing_creds.append("TOPSTEPX_API_KEY")
        else:
            print(f"   ✅ TOPSTEPX_API_KEY - Configured (length: {len(self.api_key)})")
            
        if not self.username:
            print("   ❌ TOPSTEPX_USERNAME - Not configured")
            missing_creds.append("TOPSTEPX_USERNAME")
        else:
            print(f"   ✅ TOPSTEPX_USERNAME - Configured ({self.username})")
            
        if not self.account_id:
            print("   ⚠️  TOPSTEPX_ACCOUNT_ID - Not configured (will retrieve from API)")
        else:
            print(f"   ✅ TOPSTEPX_ACCOUNT_ID - Configured ({self.account_id})")
        
        return missing_creds

    def test_authentication(self):
        """Test authentication with TopstepX API using correct endpoint"""
        print("\n🔐 AUTHENTICATION TEST:")
        
        if not self.api_key or not self.username:
            print("   ❌ Cannot test authentication - missing API key or username")
            return False
            
        # Correct TopstepX API endpoint
        url = f"{self.api_base}/api/Auth/loginKey"
        payload = {
            "userName": self.username,
            "apiKey": self.api_key
        }
        
        print(f"   🔄 POST {url}")
        print(f"   � Testing authentication for user: {self.username}")
        
        try:
            response = requests.post(
                url, 
                json=payload, 
                headers={"Content-Type": "application/json"}, 
                timeout=15
            )
            
            print(f"   � Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   📄 Response: {str(data)[:200]}...")
                
                if data.get("token"):
                    token = data["token"]
                    print("   ✅ Authentication successful!")
                    print(f"   🎫 JWT Token obtained (first 20 chars): {token[:20]}...")
                    
                    # Test account access
                    self.test_account_access(token)
                    
                    # Update .env with token if needed
                    self.update_env_with_token(token)
                    
                    return True
                else:
                    print(f"   ❌ Auth 200 but no token: {data}")
                    return False
                    
            elif response.status_code == 404:
                print("   ❌ 404: Wrong URL. Check BASE and make sure path is /api/Auth/loginKey")
                print(f"   � Attempted URL: {url}")
                return False
            else:
                print(f"   ❌ Authentication failed: {response.status_code}")
                print(f"   📄 Error: {response.text[:300]}")
                return False
                
        except Exception as e:
            print(f"   ❌ Request error: {e}")
            return False

    def test_account_access(self, token):
        """Test account information access"""
        print("\n💰 ACCOUNT ACCESS TEST:")
        
        try:
            headers = {"Authorization": f"Bearer {token}"}
            
            # Try to get account information
            account_url = f"{self.api_base}/api/Account"
            response = requests.get(account_url, headers=headers, timeout=15)
            
            print(f"   🔄 GET {account_url}")
            print(f"   📥 Status: {response.status_code}")
            
            if response.status_code == 200:
                account_data = response.json()
                print("   ✅ Account access successful!")
                
                if 'data' in account_data and account_data['data']:
                    accounts = account_data['data']
                    print(f"   📊 Found {len(accounts)} account(s):")
                    
                    primary_account_id = None
                    for i, acc in enumerate(accounts[:3]):  # Show first 3
                        acc_id = acc.get('id', 'N/A')
                        acc_name = acc.get('name', 'N/A')
                        balance = acc.get('balance', 'N/A')
                        acc_type = acc.get('type', 'N/A')
                        status = acc.get('status', 'N/A')
                        
                        print(f"      {i+1}. Account {acc_id}: {acc_name}")
                        print(f"         Type: {acc_type}, Balance: ${balance}, Status: {status}")
                        
                        # Use first account as primary if not configured
                        if i == 0 and not self.account_id:
                            primary_account_id = str(acc_id)
                            print(f"      🎯 Using as primary account: {primary_account_id}")
                        
                        # Check if this matches configured account ID
                        if str(acc_id) == str(self.account_id):
                            print(f"      ✅ Configured account ID {self.account_id} found!")
                    
                    # Update .env file with account ID if we found one
                    if primary_account_id and not self.account_id:
                        self.update_env_with_account_id(primary_account_id)
                
                return True
            else:
                print(f"   ⚠️  Account access failed: {response.status_code}")
                print(f"   📄 Response: {response.text[:200]}")
                print("   💡 This might be normal - continuing with auth validation")
                return True  # Auth was successful even if account access fails
                
        except Exception as e:
            print(f"   ⚠️  Account access error: {e}")
            print("   💡 This might be normal - continuing with auth validation")
            return True  # Auth was successful even if account access fails

    def update_env_with_token(self, token):
        """Update .env file with the JWT token"""
        try:
            env_file = Path(".env")
            if env_file.exists():
                with open(env_file, 'r') as f:
                    lines = f.readlines()
                
                # Update or add JWT token
                updated = False
                for i, line in enumerate(lines):
                    if line.startswith('TOPSTEPX_JWT='):
                        lines[i] = f'TOPSTEPX_JWT={token}\n'
                        updated = True
                        break
                
                if not updated:
                    # Add JWT token after other TopstepX credentials
                    for i, line in enumerate(lines):
                        if line.startswith('TOPSTEPX_USERNAME='):
                            lines.insert(i + 1, f'TOPSTEPX_JWT={token}\n')
                            break
                
                # Write back to file
                with open(env_file, 'w') as f:
                    f.writelines(lines)
                
                print(f"\n📝 Updated .env with JWT token (expires in ~24h)")
                
        except Exception as e:
            print(f"\n⚠️  Failed to update .env with JWT token: {e}")

    def update_env_with_account_id(self, account_id):
        """Update .env file with the discovered account ID"""
        print(f"\n📝 UPDATING .ENV FILE:")
        print(f"   Adding TOPSTEPX_ACCOUNT_ID={account_id}")
        
        try:
            env_file = Path(".env")
            if env_file.exists():
                with open(env_file, 'r') as f:
                    lines = f.readlines()
                
                # Update or add account ID
                updated = False
                for i, line in enumerate(lines):
                    if line.startswith('TOPSTEPX_ACCOUNT_ID='):
                        lines[i] = f'TOPSTEPX_ACCOUNT_ID={account_id}\n'
                        updated = True
                        break
                
                if not updated:
                    # Add account ID after other TopstepX credentials
                    for i, line in enumerate(lines):
                        if line.startswith('TOPSTEPX_USERNAME='):
                            lines.insert(i + 1, f'TOPSTEPX_ACCOUNT_ID={account_id}\n')
                            break
                
                # Write back to file
                with open(env_file, 'w') as f:
                    f.writelines(lines)
                
                print("   ✅ .env file updated successfully!")
                self.account_id = account_id
                
        except Exception as e:
            print(f"   ❌ Failed to update .env file: {e}")

    def print_next_steps(self, auth_success):
        """Print next steps based on test results"""
        print("\n" + "=" * 60)
        print("📋 AUTHENTICATION CHECKLIST RESULTS:")
        print("=" * 60)
        
        if auth_success:
            print("✅ AUTHENTICATION: PASSED")
            print("✅ API CONNECTION: WORKING") 
            print("✅ JWT TOKEN: OBTAINED")
            print("\n🎉 READY FOR NEXT STEP: Core Safety Systems")
            print("\nNext commands to run:")
            print("   dotnet build src/UnifiedOrchestrator")
            print("   dotnet run --project src/UnifiedOrchestrator -- --dry-run")
        else:
            print("❌ AUTHENTICATION: FAILED")
            print("\n🔧 REQUIRED ACTIONS:")
            print("1. Verify your .env file has correct values:")
            print("   TOPSTEPX_API_BASE=https://api.topstepx.com")
            print("   TOPSTEPX_API_KEY=your_projectx_api_key_here")
            print("   TOPSTEPX_USERNAME=your_projectx_login_email@domain.com")
            print("\n2. Make sure you're using your ProjectX login email (not Topstep login)")
            print("3. Verify API key is generated in ProjectX dashboard")
            print("4. Re-run this test: python auth_test.py")

def main():
    test = TopstepXConnectionTest()
    test.print_banner()
    
    # Check credentials configuration
    missing_creds = test.check_credentials()
    
    # Only fail if API key or username is missing
    critical_missing = [cred for cred in missing_creds if cred in ['TOPSTEPX_API_KEY', 'TOPSTEPX_USERNAME']]
    
    if critical_missing:
        print(f"\n❌ Missing critical credentials: {', '.join(critical_missing)}")
        test.print_next_steps(False)
        return False
    
    # Test authentication with correct endpoint
    auth_success = test.test_authentication()
    test.print_next_steps(auth_success)
    
    return auth_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
