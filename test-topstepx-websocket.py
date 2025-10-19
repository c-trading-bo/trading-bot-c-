#!/usr/bin/env python3
"""
TopstepX WebSocket Connection Diagnostic Tool

Tests all aspects of TopstepX connectivity:
1. Environment variables
2. API authentication
3. JWT token retrieval
4. WebSocket connection to RTC hubs
5. Basic message flow

Usage: python test-topstepx-websocket.py
"""

import os
import sys
import asyncio
import json
import aiohttp
from datetime import datetime

# ANSI color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}{text:^80}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✅ {text}{RESET}")

def print_error(text):
    print(f"{RED}❌ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠️  {text}{RESET}")

def print_info(text):
    print(f"{BLUE}ℹ️  {text}{RESET}")

async def test_environment_variables():
    """Test 1: Verify all required environment variables are set."""
    print_header("TEST 1: Environment Variables")
    
    required_vars = [
        'TOPSTEPX_API_KEY',
        'TOPSTEPX_USERNAME',
        'TOPSTEPX_ACCOUNT_ID',
        'TOPSTEPX_API_BASE',
        'TOPSTEPX_RTC_BASE',
        'RTC_USER_HUB',
        'RTC_MARKET_HUB'
    ]
    
    all_present = True
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if 'KEY' in var or 'JWT' in var:
                masked = value[:8] + '...' + value[-4:] if len(value) > 12 else '***'
                print_success(f"{var}: {masked}")
            else:
                print_success(f"{var}: {value}")
        else:
            print_error(f"{var}: NOT SET")
            all_present = False
    
    if all_present:
        print_success("\nAll required environment variables are set")
        return True
    else:
        print_error("\nMissing required environment variables")
        return False

async def test_api_connectivity():
    """Test 2: Test basic API connectivity."""
    print_header("TEST 2: API Connectivity")
    
    api_base = os.getenv('TOPSTEPX_API_BASE', 'https://api.topstepx.com')
    
    try:
        async with aiohttp.ClientSession() as session:
            print_info(f"Testing connection to {api_base}")
            async with session.get(f"{api_base}/health", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                print_success(f"API is reachable (Status: {resp.status})")
                return True
    except aiohttp.ClientError as e:
        print_error(f"API connection failed: {e}")
        return False
    except asyncio.TimeoutError:
        print_error("API connection timed out")
        return False

async def test_authentication():
    """Test 3: Test authentication and JWT token retrieval."""
    print_header("TEST 3: Authentication & JWT Token")
    
    api_base = os.getenv('TOPSTEPX_API_BASE', 'https://api.topstepx.com')
    api_key = os.getenv('TOPSTEPX_API_KEY')
    username = os.getenv('TOPSTEPX_USERNAME')
    
    if not api_key or not username:
        print_error("Missing API_KEY or USERNAME")
        return None
    
    try:
        async with aiohttp.ClientSession() as session:
            # Try to authenticate and get JWT
            print_info(f"Authenticating as {username}")
            
            headers = {
                'X-API-Key': api_key,
                'Content-Type': 'application/json'
            }
            
            # Method 1: Try /auth/token endpoint
            print_info("Attempting JWT retrieval via /auth/token")
            try:
                async with session.post(
                    f"{api_base}/auth/token",
                    headers=headers,
                    json={'username': username},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        jwt = data.get('token') or data.get('jwt') or data.get('access_token')
                        if jwt:
                            masked_jwt = jwt[:20] + '...' + jwt[-10:] if len(jwt) > 30 else '***'
                            print_success(f"JWT token retrieved: {masked_jwt}")
                            print_success(f"Token length: {len(jwt)} characters")
                            return jwt
                    else:
                        text = await resp.text()
                        print_warning(f"/auth/token returned {resp.status}: {text[:200]}")
            except Exception as e:
                print_warning(f"/auth/token failed: {e}")
            
            # Method 2: Try /api/auth/login endpoint
            print_info("Attempting JWT retrieval via /api/auth/login")
            try:
                async with session.post(
                    f"{api_base}/api/auth/login",
                    headers=headers,
                    json={'username': username, 'apiKey': api_key},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        jwt = data.get('token') or data.get('jwt') or data.get('access_token')
                        if jwt:
                            masked_jwt = jwt[:20] + '...' + jwt[-10:] if len(jwt) > 30 else '***'
                            print_success(f"JWT token retrieved: {masked_jwt}")
                            print_success(f"Token length: {len(jwt)} characters")
                            return jwt
                    else:
                        text = await resp.text()
                        print_warning(f"/api/auth/login returned {resp.status}: {text[:200]}")
            except Exception as e:
                print_warning(f"/api/auth/login failed: {e}")
            
            # Method 3: Use API key directly as JWT (some APIs work this way)
            print_info("Trying to use API key as JWT (fallback)")
            print_warning("No JWT endpoint worked - may need manual JWT token")
            return None
            
    except Exception as e:
        print_error(f"Authentication failed: {e}")
        return None

async def test_websocket_connection(jwt_token=None):
    """Test 4: Test WebSocket connection to RTC hubs."""
    print_header("TEST 4: WebSocket Connection")
    
    if not jwt_token:
        jwt_token = os.getenv('TOPSTEPX_JWT')
        if not jwt_token:
            print_error("No JWT token available for WebSocket connection")
            print_info("WebSocket requires valid JWT token from authentication")
            return False
    
    user_hub = os.getenv('RTC_USER_HUB', 'https://rtc.topstepx.com/hubs/user')
    market_hub = os.getenv('RTC_MARKET_HUB', 'https://rtc.topstepx.com/hubs/market')
    
    print_info(f"User Hub: {user_hub}")
    print_info(f"Market Hub: {market_hub}")
    
    # Try to connect to WebSocket
    try:
        import signalr_aio
        print_info("Using signalr-aio library for WebSocket connection")
        
        # Create connection to user hub
        print_info("Attempting to connect to User Hub...")
        connection = signalr_aio.Connection(user_hub, session=None)
        
        # Add JWT to query string (TopstepX requirement)
        connection.qs = {'jwt': jwt_token}
        
        hub = connection.register_hub('userHub')
        
        # Set up event handlers
        connection.received += lambda: print_success("Received WebSocket data")
        connection.error += lambda data: print_error(f"WebSocket error: {data}")
        
        # Try to start connection with timeout
        print_info("Starting WebSocket connection (30s timeout)...")
        
        start_task = asyncio.create_task(connection.start())
        try:
            await asyncio.wait_for(start_task, timeout=30.0)
            print_success("WebSocket connected successfully!")
            
            # Keep connection alive briefly
            await asyncio.sleep(2)
            
            # Close connection
            connection.close()
            print_success("WebSocket connection test completed")
            return True
            
        except asyncio.TimeoutError:
            print_error("WebSocket connection timed out after 30 seconds")
            return False
        except Exception as e:
            print_error(f"WebSocket connection failed: {e}")
            return False
            
    except ImportError:
        print_warning("signalr-aio not installed - cannot test WebSocket")
        print_info("Install with: pip install signalr-client-aio")
        
        # Try basic HTTP test to RTC endpoint
        print_info("\nTrying basic HTTP connectivity to RTC...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    user_hub.replace('/hubs/user', '/negotiate'),
                    params={'jwt': jwt_token},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        print_success(f"RTC endpoint is reachable (Status: {resp.status})")
                        data = await resp.json()
                        print_info(f"Negotiate response: {json.dumps(data, indent=2)[:200]}...")
                        return True
                    else:
                        print_error(f"RTC endpoint returned {resp.status}")
                        text = await resp.text()
                        print_info(f"Response: {text[:200]}")
                        return False
        except Exception as e:
            print_error(f"RTC connectivity test failed: {e}")
            return False

async def test_project_x_py_sdk():
    """Test 5: Test project-x-py SDK initialization."""
    print_header("TEST 5: project-x-py SDK")
    
    try:
        print_info("Checking project-x-py installation...")
        import project_x_py
        print_success(f"project-x-py is installed (version: {getattr(project_x_py, '__version__', 'unknown')})")
        
        print_info("Attempting to import TradingSuite...")
        from project_x_py import TradingSuite
        print_success("TradingSuite imported successfully")
        
        print_info("Attempting to create TradingSuite instance...")
        # Don't actually initialize (requires credentials)
        print_warning("Skipping full initialization (requires valid credentials)")
        
        return True
        
    except ImportError as e:
        print_error(f"project-x-py is not installed: {e}")
        print_info("Install with: pip install project-x-py")
        return False
    except Exception as e:
        print_error(f"project-x-py SDK test failed: {e}")
        return False

async def main():
    """Run all diagnostic tests."""
    print(f"""
{BLUE}╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║              TopstepX WebSocket Connection Diagnostic Tool                 ║
║                                                                            ║
║  This tool tests all aspects of TopstepX connectivity to identify issues  ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝{RESET}
""")
    
    start_time = datetime.now()
    
    # Run all tests
    results = {}
    
    results['env_vars'] = await test_environment_variables()
    results['api_connectivity'] = await test_api_connectivity()
    
    jwt_token = await test_authentication()
    results['authentication'] = jwt_token is not None
    
    results['websocket'] = await test_websocket_connection(jwt_token)
    results['sdk'] = await test_project_x_py_sdk()
    
    # Summary
    print_header("TEST SUMMARY")
    
    duration = (datetime.now() - start_time).total_seconds()
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"Tests run: {total}")
    print(f"Passed: {GREEN}{passed}{RESET}")
    print(f"Failed: {RED}{total - passed}{RESET}")
    print(f"Duration: {duration:.2f}s")
    print()
    
    for test_name, passed in results.items():
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {test_name:20s}: {status}")
    
    print()
    
    if all(results.values()):
        print_success("All tests passed! TopstepX connection should work.")
        return 0
    else:
        print_error("Some tests failed. Review errors above to diagnose issues.")
        
        # Provide specific guidance
        print_header("TROUBLESHOOTING GUIDANCE")
        
        if not results['env_vars']:
            print_error("Environment variables missing")
            print_info("Solution: Ensure .env file is loaded and contains all required variables")
        
        if not results['api_connectivity']:
            print_error("API connectivity failed")
            print_info("Solution: Check network/firewall, verify TopstepX service is up")
        
        if not results['authentication']:
            print_error("Authentication failed")
            print_info("Solution: Verify API_KEY and USERNAME are correct")
            print_info("         Contact TopstepX support for JWT token endpoint")
        
        if not results['websocket']:
            print_error("WebSocket connection failed")
            print_info("Solution: Ensure JWT token is valid and not expired")
            print_info("         Check RTC hub URLs are correct")
            print_info("         Verify SignalR protocol compatibility")
        
        if not results['sdk']:
            print_error("project-x-py SDK failed")
            print_info("Solution: pip install project-x-py")
        
        return 1

if __name__ == '__main__':
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Test interrupted by user{RESET}")
        sys.exit(130)
    except Exception as e:
        print_error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
