# 🏗️ Code Quality & Development Standards

## 🔒 CRITICAL: LOCKED FILES (READ FIRST!)

**BEFORE making ANY changes, read**: `.github/AI_AGENT_CONSTRAINTS.md`

**LOCKED FILES - NEVER MODIFY**:
- ❌ `.github/workflows/selfhosted-bot-run.yml` - Python/SDK setup section
- ❌ Workflow trigger configuration (branches-ignore)
- ❌ Workflow timeout settings

**If you see workflow/SDK errors**: Fix the **BOT CODE**, NOT the workflow setup.

---

## ❌ NEVER DO THESE (BUILD/CODE QUALITY)

❌ **Stub Code**: Never generate stub methods, placeholder implementations, or TODO comments in production code  
❌ **Mock Services**: Never use mock/fake services in production dependency injection container  
❌ **Fake Data**: Never generate fake data, simulated responses, or hardcoded test values in production code  
❌ **NotImplemented**: Never use `throw new NotImplementedException()` in production code paths  
❌ **Config Changes**: Never modify `Directory.Build.props`, `.editorconfig`, analyzer rule sets, or project files to bypass warnings  
❌ **Suppressions**: Never add `#pragma warning disable` or `[SuppressMessage]` without explicit approval  
❌ **Analyzer Bypasses**: Never disable `TreatWarningsAsErrors` or remove analyzer packages  
❌ **Baseline Changes**: Never "fix" the existing ~1500 analyzer warnings without explicit request  
❌ **Pattern Violations**: Never deviate from established async/await, DI, or error handling patterns  
❌ **Secret Exposure**: Never log tokens, API keys, or trading account details in code or logs  

## ✅ ALWAYS DO THESE (CODE QUALITY)

✅ **Production Ready**: Every method, service, and feature must be fully implemented before merge (no stubs/mocks/TODOs)  
✅ **Real Implementations**: Always use real APIs (TopstepX), real data sources, and complete error handling  
✅ **Complete Features**: Every function must be 100% complete with full logging and error handling  
✅ **Minimal Changes**: Make surgical, targeted fixes only - no large rewrites  
✅ **Test Everything**: Run `./dev-helper.sh analyzer-check` before every commit  
✅ **Follow Patterns**: Use existing code patterns and architectural styles  
✅ **Decimal Precision**: Use `decimal` for all monetary values and price calculations  
✅ **Proper Async**: Use `async/await` with `ConfigureAwait(false)` in libraries  
✅ **Null Safety**: Use nullable reference types and null-conditional operators  
✅ **Logging**: Use structured logging with appropriate log levels (Debug/Info/Warning/Error/Critical)  

## 🔧 Build & Quality Standards

### Compiler Requirements
- **Zero New Warnings**: Build must pass `dotnet build -warnaserror` with no new analyzer violations
- **Existing Baseline**: Respect the documented ~1500 existing warnings - do not attempt to fix them
- **Analyzer Compliance**: `TreatWarningsAsErrors=true` must remain enabled in Directory.Build.props
- **No Shortcuts**: Zero suppressions or config modifications to bypass quality gates

### Code Quality Gates
- **Test Coverage**: All changes must pass existing test suite without modification
- **Performance**: No degradation in latency-critical trading operations (< 10ms order execution)
- **Pattern Consistency**: Follow existing async/await, DI, and error handling patterns
- **Security**: No exposure of credentials, tokens, or trading account information

### Data Type Standards
- **Money**: Always use `decimal` for prices, PnL, account balances, risk calculations
- **Timestamps**: Use `DateTimeOffset` for all time values (handles time zones correctly)
- **IDs**: Use `string` for order IDs, account IDs, symbol IDs (broker-agnostic)
- **Quantities**: Use `int` for contracts/shares, `decimal` for fractional shares (crypto/stocks)

### Production-Ready Code Requirements
**EVERY piece of code must be production-ready before merge:**

❌ **PROHIBITED in Production Code**:
- Stub methods returning hardcoded values
- `throw new NotImplementedException()`
- `// TODO:` comments in production paths
- Mock services in production DI container
- Fake data generators or simulated responses
- Hardcoded test values in business logic
- Placeholder configuration
- Incomplete error handling

✅ **REQUIRED in All Code**:
- Complete implementation of every method
- Real API connections (TopstepX, not mocks)
- Real data sources (live market data)
- Full error handling (try/catch with logging)
- Comprehensive logging for critical operations
- All configuration from .env or appsettings.json
- All dependencies properly resolved
- Production-grade quality (no temporary workarounds)

**Example - WRONG (Stub)**:
```csharp
public async Task<decimal> GetAccountEquityAsync() {
    // TODO: Implement real query
    return 100000m; // Fake value
}
```

**Example - CORRECT (Real)**:
```csharp
public async Task<decimal> GetAccountEquityAsync() {
    try {
        var account = await _topstepxAdapter.GetAccountInfoAsync();
        if (account == null) {
            _logger.LogError("Failed to retrieve account info");
            throw new InvalidOperationException("Account info unavailable");
        }
        _logger.LogDebug("Account equity: {Equity}", account.Equity);
        return account.Equity;
    }
    catch (Exception ex) {
        _logger.LogError(ex, "Error retrieving account equity");
        throw;
    }
}
```

## 📋 Development Workflow

### 1. Setup & Validation
```bash
./dev-helper.sh setup              # Install dependencies, verify environment
./validate-agent-setup.sh          # Validate dev environment configuration
./dev-helper.sh build              # Must pass with existing warnings only
```

### 2. Code Implementation
```bash
# Make minimal, surgical changes only
# Follow existing code patterns exactly
# Use decimal for all monetary calculations
# Implement proper async/await patterns
# Add comprehensive logging at key decision points
```

### 3. Quality Validation
```bash
./dev-helper.sh build              # Check for compilation errors
./dev-helper.sh analyzer-check     # Verify no new warnings introduced
./dev-helper.sh test               # Ensure all tests pass
dotnet format --verify-no-changes  # Verify code formatting compliance
```

### 4. Pre-Commit Checklist
- [ ] No new analyzer warnings introduced
- [ ] All tests pass (unit + integration)
- [ ] Code follows existing patterns
- [ ] Logging added for key operations
- [ ] No secrets or credentials in code
- [ ] Decimal types used for all money values

## 🎯 Key Development Files

### Core Projects
- `src/UnifiedOrchestrator/` - Main trading orchestration and startup
- `src/BotCore/` - Core services, interfaces, and dependency injection
- `src/TopstepAuthAgent/` - TopstepX API integration layer
- `src/Safety/` - Production safety mechanisms and health monitoring
- `src/adapters/` - Python bridge to TopstepX SDK

### Configuration Files
- `.env` - Environment configuration (copy from `.env.example`, never commit)
- `Directory.Build.props` - **DO NOT MODIFY** - Contains analyzer rules and build settings
- `appsettings.*.json` - Application settings for different environments
- `strategies-enabled.json` - Strategy configuration

### Helper Scripts
- `./dev-helper.sh` - Development automation (build, test, analyze)
- `./validate-agent-setup.sh` - Environment validation
- `./test-production-guardrails.sh` - Safety mechanism verification

## 📊 Quality Metrics

| Requirement | Validation Command | Status |
|-------------|-------------------|---------|
| Zero New Warnings | `./dev-helper.sh analyzer-check` | ✅ Required |
| Test Compliance | `./dev-helper.sh test` | ✅ Required |
| Code Formatting | `dotnet format --verify-no-changes` | ✅ Required |
| Build Success | `./dev-helper.sh build` | ✅ Required |

## � Logging Standards

### Log Levels
- **Debug**: Detailed diagnostic information (market data ticks, internal state)
- **Information**: Key business events (order placed, position opened, strategy activated)
- **Warning**: Unexpected but recoverable situations (API retry, missing optional data)
- **Error**: Failures requiring attention (order rejection, API connection lost)
- **Critical**: System-wide failures (safety system failure, data corruption)

### Structured Logging Example
```csharp
_logger.LogInformation(
    "Order submitted: {Symbol} {Side} {Quantity} @ {Price} (OrderId: {OrderId})",
    symbol, side, quantity, price, orderId
);
```

### What to Log
✅ **Log**: Order submissions, fills, position changes, strategy signals, API calls, safety triggers  
✅ **Log**: Performance metrics, latency measurements, error conditions, configuration changes  
❌ **Never Log**: API keys, tokens, passwords, account numbers, personal information  

## 🔍 Code Review Checklist

Before submitting changes, verify:
- [ ] All money values use `decimal` type
- [ ] All async methods use `ConfigureAwait(false)` in library code
- [ ] All API calls have timeout and retry logic
- [ ] All database queries use parameterized statements
- [ ] All user inputs are validated before processing
- [ ] All file operations handle missing/corrupt files gracefully
- [ ] All exceptions are logged with sufficient context
- [ ] All critical paths have performance logging
- [ ] All new code follows existing architectural patterns
- [ ] All configuration uses strongly-typed IOptions<T> pattern

Remember: **Code quality prevents production issues. Take time to do it right.**
