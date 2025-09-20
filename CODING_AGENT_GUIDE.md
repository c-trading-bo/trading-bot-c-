# Coding Agent Quick Start Guide

This guide helps coding agents get up and running quickly with the trading bot repository.

## 🚀 Quick Start

### Build & Test
```bash
# Restore dependencies
dotnet restore

# Build solution (expect analyzer warnings - don't fix unless asked)
dotnet build --no-restore

# Run tests
dotnet test --no-build --verbosity normal

# Run specific tests
dotnet test tests/Unit/MLRLAuditTests.csproj
```

### Key Entry Points

1. **Main Trading Application**: `src/UnifiedOrchestrator/` - Primary orchestration logic
2. **Core Trading Logic**: `src/BotCore/` - Core trading engine and services  
3. **TopstepX Integration**: `src/TopstepAuthAgent/` - API client for TopstepX
4. **ML/RL Components**: `src/RLAgent/`, `src/ML/` - Machine learning and reinforcement learning
5. **Strategy Layer**: `src/Strategies/`, `src/StrategyAgent/` - Trading strategies

### Configuration
- Environment settings: `.env` (copy from `.env.example`)
- Production config: `src/BotCore/Services/ProductionConfigurationService.cs`
- App settings: `appsettings.*.json` files

## 🎯 Agent Guidelines

### Development Workflow
1. **Always run `dotnet restore` first** when starting
2. **Build before making changes** to understand current state
3. **Use minimal changes** - surgical fixes only
4. **Test frequently** with `dotnet test` 
5. **Follow existing patterns** - don't introduce new architectures

### Code Style
- **Async-first**: Use `async Task` and `ConfigureAwait(false)`
- **Decimal for money**: Always use `decimal` for prices/quantities
- **Structured logging**: Use ILogger with structured messages
- **File-scoped namespaces**: Use `namespace Foo.Bar;` syntax

### Trading-Specific Rules
- **ES/MES tick size**: Round to 0.25 increments using `Px.RoundToTick()`
- **DRY_RUN by default**: Never execute real trades without explicit flag
- **Order evidence required**: Must have orderId + fill confirmation
- **No secrets in logs**: Never log tokens, passwords, or API keys

### Key Files to Understand
- `.github/copilot-instructions.md` - Comprehensive instructions
- `src/BotCore/Services/` - Core services and configurations
- `Directory.Build.props` - Global build settings and analyzer rules

### Common Tasks

#### Adding a new service
1. Create in appropriate `src/` subfolder
2. Add project reference to solution
3. Register in DI container
4. Add interface to `src/Abstractions/` if needed

#### TopstepX API integration
- Use `src/TopstepAuthAgent/TopstepAuthAgent.cs` patterns
- Follow SignalR hub patterns in copilot instructions
- All API calls need proper error handling and retries

#### ML/RL modifications  
- Main agents in `src/RLAgent/` and `src/ML/`
- Use existing ONNX model patterns
- Respect training/inference separation

## ⚠️ Important Warnings

### Don't Touch These
- `.env` files (except for new setup)
- Production secrets and credentials
- `kill.txt` safety mechanism
- Core auth/login code
- Risk management controls
- CI/CD workflows without explicit need

### Build Warnings
- **Analyzer warnings are expected** - don't fix unless specifically asked
- **Build may fail on first run** - this is normal due to strict analyzers
- **Focus on functional changes** not code quality fixes

### Safety First
- **VPN/Remote trading blocked** - local development only
- **Multiple safety switches** - DRY_RUN, kill.txt, risk limits
- **Order validation required** - never bypass order proof requirements

## 🔍 Debugging

### Logs Location
- Console output for development
- Structured JSON logs in production
- Alert system logs to email/Slack

### Common Issues
1. **Build failures**: Usually analyzer warnings, can ignore unless breaking
2. **Missing dependencies**: Run `dotnet restore`
3. **Auth failures**: Check `.env` file configuration
4. **Port conflicts**: Default port 5050, check if already in use

### Health Checks
```bash
# Test health endpoints
curl http://localhost:5050/healthz
curl http://localhost:5050/healthz/mode
```

## 📚 Reference

- **TopstepX API Docs**: Live endpoints in copilot instructions
- **SignalR Patterns**: Canonical implementations in instructions
- **Risk/Pricing Logic**: `Px` class helpers for tick rounding
- **Order Flow**: Signal → Intent → Place → Confirm pattern

---

*This guide is for coding agents. For human developers, see the main README.md and project documentation.*