# SignalR Production-Ready Connection Stability Testing

This document describes how to test and validate the production-ready SignalR connection stability improvements implemented in this repository.

## Changes Implemented

### 1. Startup Sequencing
- **Before**: Hubs started before JWT validation
- **After**: `WaitForJwtReadinessAsync()` waits up to 45 seconds for valid JWT before any hub connection
- **File**: `src/UnifiedOrchestrator/Services/SignalRConnectionManager.cs`

### 2. Token Refresh Policy  
- **Before**: Skip restart logic could leave hubs in bad state
- **After**: `OnTokenRefreshed()` handler immediately restarts disconnected hubs on fresh token
- **File**: `src/UnifiedOrchestrator/Services/SignalRConnectionManager.cs`

### 3. Transport & Negotiation
- **Before**: `Transports = WebSockets | LongPolling | SSE`, `SkipNegotiation = false`
- **After**: `Transports = WebSockets`, `SkipNegotiation = true` (WebSockets only)
- **Files**: `src/UnifiedOrchestrator/Services/SignalRConnectionManager.cs`, `src/BotCore/MarketHubClient.cs`

### 4. Legacy Handler Override
- **Before**: `Environment.SetEnvironmentVariable("DOTNET_SYSTEM_NET_HTTP_USESOCKETSHTTPHANDLER", "false")`
- **After**: Removed override, uses default SocketsHttpHandler
- **File**: `src/UnifiedOrchestrator/Services/SignalRConnectionManager.cs`

### 5. AccessTokenProvider Hygiene
- **Before**: Static token provider
- **After**: Dynamic provider that always returns freshest token with Bearer prefix stripping
- **File**: `src/UnifiedOrchestrator/Services/SignalRConnectionManager.cs`

### 6. Connection Timing
- **Before**: No explicit delay for token readiness
- **After**: Structured 30-45 second delay with logging
- **File**: `src/UnifiedOrchestrator/Services/SignalRConnectionManager.cs`

### 7. Clock Validation
- **Before**: No JWT timing validation
- **After**: `ValidateJwtTokenAsync()` checks exp/nbf claims and system clock
- **File**: `src/UnifiedOrchestrator/Services/SignalRConnectionManager.cs`

### 8. Credential Path Cleanup
- **Before**: Used `~/.topstepx/` for credential storage
- **After**: Uses Roaming AppData with cleanup of legacy `.topstepx` paths
- **File**: `src/Infrastructure.TopstepX/TopstepXCredentialManager.cs`

## Testing the Implementation

### Quick Test (Basic Validation)

```bash
cd TestSignalR
dotnet run
```

This will test:
- JWT token discovery and validation
- REST API authentication  
- Production-ready SignalR connections with:
  - JWT readiness waiting
  - WebSockets-only transport
  - Fresh token providers
  - Extended stability validation
  - Immediate subscription success

### Extended Test (10+ Minutes Stability)

```bash
cd TestSignalR
dotnet run -- --extended
```

This will run the full 12-minute stability test to validate:
- No "Normal closure" errors
- No "InvokeCoreAsync cannot be called" errors  
- Continuous ES/NQ market data reception
- Continuous order/trade events
- Connection stability under load

### Manual Testing with Real Environment

1. Set up TopstepX credentials:
   ```bash
   export TOPSTEPX_USERNAME="your_username"
   export TOPSTEPX_API_KEY="your_api_key"
   ```

2. Or place JWT token in:
   - `state/auth_token.txt`
   - `state/access_token.txt` 
   - `state/jwt_token.txt`
   - Or `TOPSTEPX_JWT` environment variable

3. Run the production system:
   ```bash
   cd src/UnifiedOrchestrator
   dotnet run
   ```

4. Monitor logs for:
   - ✅ "JWT ready before hub connection attempts"
   - ✅ "Both hubs connect with valid ConnectionId" 
   - ✅ "Subscriptions succeed immediately after connect"
   - ❌ No "Normal closure" or "InvokeCoreAsync" errors

## Expected Startup Log Sequence

```
[INFO] SignalRManager: Waiting for JWT token readiness before establishing hub connections...
[INFO] SignalRManager: JWT token ready after 2.1 seconds
[INFO] SignalRManager: Using validated JWT token for User Hub: length=1234, has_dots=3
[INFO] SignalRManager: Starting User Hub connection with enhanced stability validation...
[INFO] SignalRManager: User Hub connection started successfully. State: Connected
[INFO] SignalRManager: User Hub stability check 1/3: ✅ State: Connected, ID: abc123
[INFO] SignalRManager: User Hub stability check 2/3: ✅ State: Connected, ID: abc123  
[INFO] SignalRManager: User Hub stability check 3/3: ✅ State: Connected, ID: abc123
[INFO] SignalRManager: User Hub connection validated after extended checks
[INFO] SignalRManager: User Hub connection established and confirmed ready
```

## Acceptance Criteria Validation

### ✅ Startup logs show JWT ready before any hub connection attempts
- Look for "JWT token ready after X seconds" before any "Starting...Hub connection"

### ✅ Both hubs connect with valid ConnectionId and remain connected  
- Look for "State: Connected, ID: [non-null-value]" in stability checks

### ✅ Subscriptions succeed immediately after connect
- Look for "subscription successful" messages immediately after connection

### ✅ No "Normal closure" or "InvokeCoreAsync cannot be called" errors
- Monitor error logs during extended test - should be clean

### ✅ 10+ minutes continuous ES/NQ market data and order/trade events without disconnect
- Run extended test and verify event counters increase throughout duration

## Production Environment Variables

```bash
# Authentication
TOPSTEPX_USERNAME=your_username
TOPSTEPX_API_KEY=your_api_key
TOPSTEPX_JWT=your_jwt_token
TOPSTEPX_ACCOUNT_ID=your_account_id

# Connection Configuration  
RTC_USER_HUB=https://rtc.topstepx.com/hubs/user
RTC_MARKET_HUB=https://rtc.topstepx.com/hubs/market
TOPSTEPX_API_BASE=https://api.topstepx.com

# Credential Storage
TRADING_CREDENTIALS_PATH=/path/to/roaming/appdata/TradingBot

# Debugging
ENABLE_SIGNALR_DEBUG=true
SIGNALR_LOG_LEVEL=Information
```

## Troubleshooting

### JWT Token Issues
- **"JWT token readiness timeout"**: Check token source and validity
- **"JWT token expired"**: Refresh token or check system clock
- **"JWT token not yet valid"**: Check system clock synchronization

### Connection Issues  
- **"Connection failed or no ConnectionId"**: Check network/firewall for WebSockets
- **"SSL certificate validation failed"**: Check cert validation settings
- **"Subscription failed"**: Check account ID and permissions

### Performance Issues
- **High CPU during connection**: Check if using SocketsHttpHandler (legacy override removed)
- **Memory leaks**: Check connection disposal in logs
- **Frequent reconnections**: Check JWT expiry and refresh timing

## Files Modified

1. `src/UnifiedOrchestrator/Services/SignalRConnectionManager.cs` - Main production fixes
2. `src/Infrastructure.TopstepX/TopstepXCredentialManager.cs` - Credential path cleanup  
3. `TestSignalR/Program.cs` - Enhanced test application

## Integration with Existing Code

The changes are backward compatible and integrate with:
- `CentralizedTokenProvider` for token management
- `AutoTopstepXLoginService` for authentication flow
- Existing hub event handlers and subscriptions
- All safety and risk management systems

The production-ready SignalR system maintains full compatibility with existing trading logic while providing enhanced stability and reliability.