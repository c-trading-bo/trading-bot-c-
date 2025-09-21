# 🛡️ Production Trading Bot Guardrails

## ❌ NEVER DO THESE (WILL BREAK PRODUCTION)

❌ **Config Changes**: Never modify `Directory.Build.props`, `.editorconfig`, analyzer rule sets, or project files to bypass warnings  
❌ **Suppressions**: Never add `#pragma warning disable` or `[SuppressMessage]` without explicit approval  
❌ **Live API Calls**: Never connect to live trading APIs from CI/cloud environments  
❌ **Analyzer Bypasses**: Never disable `TreatWarningsAsErrors` or remove analyzer packages  
❌ **Secret Exposure**: Never log tokens, API keys, or trading account details  
❌ **VPN Trading**: Never execute trades from VPN, VPS, or remote desktop connections  
❌ **Baseline Changes**: Never "fix" the existing ~1500 analyzer warnings without explicit request  
❌ **Order Bypasses**: Never claim order fills without orderId + fill event proof  
❌ **Risk Bypasses**: Never skip ES/MES tick rounding (0.25) or risk validation (≤ 0)  
❌ **Safety Bypasses**: Never disable DRY_RUN mode, kill.txt monitoring, or production guardrails  

## ✅ ALWAYS DO THESE (PRODUCTION REQUIREMENTS)

✅ **Minimal Changes**: Make surgical, targeted fixes only - no large rewrites  
✅ **Test Everything**: Run `./dev-helper.sh analyzer-check` before every commit  
✅ **Follow Patterns**: Use existing code patterns and architectural styles  
✅ **Verify Safety**: Ensure all production guardrails remain functional  
✅ **Decimal Precision**: Use `decimal` for all monetary values and price calculations  
✅ **Proper Async**: Use `async/await` with `ConfigureAwait(false)` in libraries  
✅ **Order Evidence**: Require orderId + GatewayUserTrade event before claiming fills  
✅ **Tick Rounding**: Round ES/MES prices to 0.25 using `Px.RoundToTick()`  
✅ **Risk Validation**: Reject trades with risk ≤ 0 using proper R-multiple calculation  
✅ **DRY_RUN Default**: Default to simulation mode unless explicitly enabled for live trading  

## 🔒 Core Production Requirements

### Build & Quality Standards
- **Zero New Warnings**: Build must pass `dotnet build -warnaserror` with no new analyzer violations
- **Existing Baseline**: Respect the documented ~1500 existing warnings - do not attempt to fix them
- **Test Coverage**: All changes must pass existing test suite without modification
- **Performance**: No degradation in latency-critical trading operations

### Trading Safety Enforcement
- **Kill Switch**: `kill.txt` file monitoring must force DRY_RUN mode automatically
- **Order Proof**: Orders require orderId return + fill event confirmation
- **Price Precision**: ES/MES tick size compliance (0.25 increments)
- **Risk Management**: Risk calculations must validate > 0 before execution
- **Environment Isolation**: Local development only - no remote trading

### Code Quality Gates
- **Analyzer Compliance**: `TreatWarningsAsErrors=true` maintained
- **No Shortcuts**: Zero suppressions or config modifications to bypass quality gates
- **Pattern Consistency**: Follow existing async/await, DI, and error handling patterns
- **Security**: No exposure of credentials, tokens, or trading account information

## 📋 Development Workflow

### 1. Setup & Validation
```bash
./dev-helper.sh setup
./validate-agent-setup.sh
./dev-helper.sh build  # Must pass with existing warnings only
```

### 2. Change Implementation
- Make minimal, surgical changes only
- Follow existing code patterns exactly
- Use `decimal` for all monetary calculations
- Implement proper async/await patterns

### 3. Quality Validation
```bash
./dev-helper.sh build          # Check for compilation errors
./dev-helper.sh analyzer-check # Verify no new warnings
./dev-helper.sh test           # Ensure tests pass
./dev-helper.sh riskcheck      # Validate trading constants
```

### 4. Production Safety Verification
- Confirm all guardrails remain functional
- Verify DRY_RUN mode compliance
- Test kill switch functionality
- Validate order evidence requirements

## 🎯 Entry Points & Key Files

### Core Trading Components
- `src/UnifiedOrchestrator/` - Main trading orchestration
- `src/BotCore/Services/` - Core services and dependency injection
- `src/TopstepAuthAgent/` - API integration layer
- `src/Safety/` - Production safety mechanisms

### Configuration & Environment
- `.env` - Environment configuration (copy from `.env.example`)
- `Directory.Build.props` - **DO NOT MODIFY** - Contains analyzer rules
- `kill.txt` - Emergency stop mechanism (creates DRY_RUN)

### Helper Scripts
- `./dev-helper.sh` - Development automation
- `./validate-agent-setup.sh` - Environment validation
- `./verify-core-guardrails.sh` - Safety mechanism verification

## 📊 Success Metrics

| Requirement | Validation Method | Status |
|-------------|------------------|---------|
| Zero New Warnings | `./dev-helper.sh analyzer-check` | ✅ Required |
| Test Compliance | `./dev-helper.sh test` | ✅ Required |
| Safety Guardrails | `./verify-core-guardrails.sh` | ✅ Required |
| Risk Validation | `./dev-helper.sh riskcheck` | ✅ Required |

## 🚨 Emergency Procedures

If production safety is compromised:
1. **Immediate**: Create `kill.txt` to force DRY_RUN mode
2. **Verify**: Run `./verify-core-guardrails.sh` to check all safety mechanisms
3. **Isolate**: Disconnect from live trading environments
4. **Audit**: Review all recent changes for compliance violations

Remember: **Production trading safety is non-negotiable. When in doubt, choose the safer option.**
