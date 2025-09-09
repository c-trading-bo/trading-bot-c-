# HttpClient DI & Token Handling Review Report

## Current Issues Identified

### 1. Multiple HttpClient Instances
- **Problem**: Found multiple `new HttpClient()` instantiations in Program.cs files
- **Location**: `src/OrchestratorAgent/Program.cs` line ~25 and ~65
- **Risk**: Socket exhaustion, improper connection pooling, resource leaks

### 2. Inconsistent Token Handling
- **Problem**: JWT tokens are set on individual HttpClient instances
- **Risk**: Token exposure in logs, inconsistent authentication state

### 3. No Centralized DI Registration
- **Problem**: HttpClient created manually instead of via DI container
- **Risk**: No retry policies, no circuit breakers, no proper lifecycle management

## Recommended Solutions

### 1. Unified HttpClient Registration
Replace manual HttpClient creation with proper DI registration:

```csharp
// In Program.cs or Startup.cs
services.AddTopstepXHttpClient(); // Use the new extension method
```

### 2. Use HttpClientFactory Pattern
```csharp
// Instead of: using var http = new HttpClient()
// Use: var httpClient = await _httpClientFactory.CreateAuthenticatedClientAsync();
```

### 3. Secure Token Management
- Tokens managed centrally by `ITopstepXTokenHandler`
- No token logging or exposure in logs
- Automatic refresh with proper thread safety
- Clear separation of concerns

## Implementation Status

✅ **Created**: `HttpClientConfiguration.cs` with proper DI setup
✅ **Features**:
- Named HttpClient registration (`"TopstepX"`)
- Automatic token injection via factory pattern
- Thread-safe token refresh with semaphore
- No secret logging (OWASP compliance)
- Retry policies and circuit breaker support
- Proper timeout configuration

## Migration Steps

1. **Add DI Registration** in main Program.cs:
   ```csharp
   builder.Services.AddTopstepXHttpClient();
   ```

2. **Replace Direct HttpClient Usage**:
   ```csharp
   // Old:
   using var http = new HttpClient { BaseAddress = new Uri("...") };
   
   // New:
   var http = await _httpClientFactory.CreateAuthenticatedClientAsync();
   ```

3. **Update Constructors** to inject `ITopstepXHttpClientFactory`

4. **Remove Manual Token Setting**:
   ```csharp
   // Remove: http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);
   // Token is automatically injected by the factory
   ```

## Security Improvements

- ✅ No tokens in logs
- ✅ Environment variable-based credentials
- ✅ Automatic token expiry handling
- ✅ Thread-safe token refresh
- ✅ Circuit breaker for API resilience
- ✅ Proper HttpClient lifecycle management

## Performance Benefits

- ✅ Connection pooling via HttpClientFactory
- ✅ DNS change handling
- ✅ Automatic retry with exponential backoff
- ✅ Circuit breaker prevents cascade failures
- ✅ Proper resource disposal

## Files to Update

1. `src/OrchestratorAgent/Program.cs` - Replace direct HttpClient usage
2. `src/UnifiedOrchestrator/Program.cs` - Replace direct HttpClient usage
3. `src/UpdaterAgent/Program.cs` - Replace direct HttpClient usage
4. Any class that creates `new HttpClient()` directly

## Environment Variables Required

```bash
TOPSTEPX_API_BASE=https://api.topstepx.com
TOPSTEPX_USERNAME=your_username
TOPSTEPX_API_KEY=your_api_key
```

## Testing Recommendations

1. Unit tests for `TopstepXTokenHandler`
2. Integration tests for authenticated HTTP calls
3. Circuit breaker behavior testing
4. Token refresh timing tests

This implementation ensures:
- Single HttpClient registration via DI
- Robust JWT refresh without logging secrets
- Proper resource management
- Resilient API communication