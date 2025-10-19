# TopstepX Trading Bot - Enhanced Multi-Brain System with Full Auto-Promotion

## 🚀 **ACTIVE SYSTEM: UnifiedOrchestrator with Production-Ready Enhanced Learning**

**⚠️ IMPORTANT: Only use `src/UnifiedOrchestrator` - all other entry points are disabled to prevent conflicts**

---

## 🏗️ **Architecture Split Analysis: Live Bot + Trainer/Gym Separation**

**Want to simplify your bot by separating live trading from learning?** We've analyzed your entire codebase (612 C# files, ~150k LOC) and created a comprehensive roadmap:

📋 **[Quick Reference](ARCHITECTURE_SPLIT_QUICK_REFERENCE.md)** ← Start here for navigation
📊 **[Executive Summary](ARCHITECTURE_SPLIT_EXECUTIVE_SUMMARY.md)** ← 10 min read for decision makers
📐 **[Visual Diagrams](ARCHITECTURE_SPLIT_DIAGRAMS.md)** ← 20 min read for architects
📖 **[Full Analysis](ARCHITECTURE_SPLIT_ANALYSIS.md)** ← 60 min read for developers

**TL;DR**: Your codebase is 60-70% ready for a clean split. Effort: 150 hours (3-4 weeks). Risk: Medium. Logic preserved: 100%.

---

## 🤖 **NEW: Launch Bot with GitHub Actions (No Local Setup Required!)**

**Want to run the bot without any local installation? Use GitHub-hosted workflows!**

🚀 **[Quick Start: GitHub Workflows](QUICK_START_GITHUB_WORKFLOWS.md)** ← Launch in 3 steps

📚 **[Complete Workflow Guide](.github/workflows/README-GITHUB-HOSTED-WORKFLOWS.md)** ← Full documentation

**Available Workflows:**
- **Bot Launch - GitHub Hosted**: Primary workflow with configurable runtime (1-60 min)
- **Bot Launch Diagnostics**: Advanced diagnostics with detailed logging
- **Bot Execution Test**: Quick 5-minute test run

**Features:**
- ✅ No self-hosted runner needed - runs on GitHub infrastructure
- ✅ Full TopstepX API connectivity
- ✅ Real-time log viewing in GitHub Actions
- ✅ Downloadable artifacts (logs, metrics, system info)
- ✅ DRY_RUN mode for safe paper trading
- ✅ Complete environment setup automated

**Requirements:** Set TopstepX credentials as GitHub secrets (see Quick Start guide)

---

## 🤖 **Real-Time Debugging with GitHub Copilot**

**Looking to launch your bot and have Copilot help you debug it in real-time?**

📖 **[SOLUTION SUMMARY](SOLUTION_SUMMARY_COPILOT_DEBUGGING.md)** ← Complete overview of all options

📖 **[Quick Start Guide](QUICK_START_COPILOT.md)** ← TL;DR version to get started fast

📚 **[Complete Debugging Guide](COPILOT_REAL_TIME_DEBUGGING_GUIDE.md)** ← Comprehensive reference

📝 **[Example Walkthrough](COPILOT_DEBUGGING_EXAMPLE.md)** ← Real-world scenarios

**Quick commands:**
- **GitHub Actions**: Use "🚀 Bot Launch - GitHub Hosted" workflow (recommended)
- **Windows**: `.\quick-launch.ps1` or `.\launch-bot-diagnostic.ps1`
- **VS Code**: Copy `.vscode-template/*` to `.vscode/`, then press `F5`

---

### 🧠 **Enhanced Multi-Brain Architecture with Auto-Promotion**
- **Primary Decision Engine**: EnhancedTradingBrainIntegration with UnifiedTradingBrain
- **ML Algorithms**: Neural UCB (627 lines) + CVaR-PPO (1,026 lines) + LSTM
- **Auto-Promotion System**: CloudRlTrainerV2 with manifest-based model updates
- **Canary Monitoring**: Automatic rollback with performance thresholds
- **Hot-Reload**: ONNX session swapping without restart
- **Cloud Integration**: 30 GitHub workflows + local/remote model discovery
- **Production Services**: 12+ enterprise-grade services with full monitoring

### ✅ **Production-Ready Features**
- **Zero Compilation Errors**: Clean build with all stubs/placeholders completed
- **Full Auto-Promotion**: Manifest-based model updates with atomic swaps
- **Enterprise Error Handling**: Circuit breakers, retries, graceful degradation
- **Real-time Monitoring**: Health checks, performance metrics, canary watchdog
- **Secure Configuration**: Environment-based settings with credential protection
- **Complete ML/RL Integration**: All algorithms active in unified decision flow
- **Bootstrap System**: Idempotent directory/file creation with sample configurations

### 🎯 **Auto-Enabled Features (Production Ready)**
- **Enhanced Learning**: Auto-starts with light (60min) and intensive (15min) modes
- **Adaptive Intelligence**: Runtime parameter adjustment based on performance
- **Model Registry**: Hot-reload notifications for brain updates
- **Canary Watchdog**: Automatic rollback on performance degradation
- **Historical Data Providers**: Hierarchical fallback (Features → Quotes → TopstepX)
- **Market Hours Enforcement**: ET maintenance, Sunday curb, CME holidays

## 🎯 **How to Run the Production System**

```bash
# Start the fully autonomous enhanced learning system
cd src/UnifiedOrchestrator
dotnet run

# System auto-creates: state/, datasets/, artifacts/, manifests/, config/
# Enhanced learning starts automatically
# Model promotion enabled by default (PROMOTE_TUNER=1)
# Live trading remains manual (LIVE_ORDERS=0)
```

### ⚡ **What Happens at Launch**
1. **Bootstrap**: Creates all required directories and sample configurations
2. **Enhanced Learning**: Starts light learning (market hours) and intensive (closed)
3. **Model Discovery**: Scans artifacts/, registry, and GitHub for available models
4. **Manifest Polling**: Checks manifests/manifest.json every 15 minutes for updates
5. **Auto-Promotion**: Downloads, verifies, and atomically swaps new models
6. **Canary Monitoring**: Tracks performance for 100 decisions or 30 minutes
7. **Hot-Reload**: Updates brain with new ONNX sessions without restart

### 🛡️ **Production Safety Preserved**

This implementation follows strict production safety guidelines:

- **No modifications** to `.editorconfig`, `Directory.Build.props`, or analyzer packages
- **Surgical changes** with minimal code modifications
- **Append-only service registration** - no removal or reordering of existing services
- **Existing guardrails preserved** - kill switch, DRY_RUN precedence, order evidence requirements
- **Live trading disabled by default** - `LIVE_ORDERS=0`, `INSTANT_ALLOW_LIVE=0` (manual activation required)
- **Model promotion enabled** - `PROMOTE_TUNER=1` (auto-learning, not live trading)

### 🔧 **Production-Ready Components Status**

✅ **All Components Fully Implemented (No Stubs/Placeholders):**

| Component | Status | Auto-Enabled | Manual Override |
|-----------|---------|--------------|----------------|
| **CloudRlTrainerV2** | ✅ Production Ready | `PROMOTE_TUNER=1` | Set to `0` to disable |
| **EnhancedBacktestLearningService** | ✅ Production Ready | `ENHANCED_LEARNING_ENABLED=1` | Set to `0` to disable |
| **CanaryWatchdog** | ✅ Production Ready | `CANARY_WATCHDOG_ENABLED=1` | Set to `0` to disable |
| **ModelRegistry** | ✅ Production Ready | `MODEL_REGISTRY_ENABLED=1` | Set to `0` to disable |
| **AdaptiveIntelligence** | ✅ Production Ready | `ADAPTIVE_INTELLIGENCE_ENABLED=1` | Set to `0` to disable |
| **MarketHoursService** | ✅ Production Ready | Always enabled | N/A |
| **HistoricalDataProviders** | ✅ Production Ready | Always enabled | N/A |
| **Bootstrap System** | ✅ Production Ready | Always enabled | N/A |

### 🚀 **Live Trading Controls (Manual Activation Required)**

| Control | Default Value | Description |
|---------|---------------|-------------|
| `LIVE_ORDERS` | `0` | Must be set to `1` to enable live order placement |
| `INSTANT_ALLOW_LIVE` | `0` | Must be set to `1` to bypass canary safety |
| `ALLOW_TOPSTEP_LIVE` | `0` | Must be set to `1` to enable TopstepX live trading |

**📋 To Enable Live Trading (Manual Process):**
```bash
# All three must be set to 1 for live trading
export LIVE_ORDERS=1
export INSTANT_ALLOW_LIVE=1  # Only if bypassing canary
export ALLOW_TOPSTEP_LIVE=1
```

### ❌ **Disabled Systems (DO NOT USE)**
- ~~`src/OrchestratorAgent`~~ - Shows warning, redirects to UnifiedOrchestrator
- ~~`app/TradingBot`~~ - Shows warning, redirects to UnifiedOrchestrator  
- ~~`SimpleBot`~~ - Shows warning, redirects to UnifiedOrchestrator

## 🛡️ **Trading Safeguards**
- **DRY_RUN Mode**: Default safe operation without live orders
- **Risk Limits**: Daily loss, position size, and drawdown enforcement
- **Emergency Shutdown**: Automatic trading halt on critical failures
- **Order Verification**: No fills without proof from TopstepX API
- **Health Monitoring**: Real-time system status with component tracking

### 📡 **TopstepX Integration**
- **REST API**: https://api.topstepx.com order placement and verification
- **SDK Adapter**: Python bridge for order/trade updates  
- **Market Data**: TopstepX SDK for real-time market data
- **Authentication**: Bearer token security with environment variables

## Quick Start

### Prerequisites
- .NET 8.0 SDK
- TopstepX API credentials
- Valid account configuration

### Build & Run
```bash
# Build the solution
dotnet build

# Run the trading bot (verified working)
dotnet run --project SimpleBot/SimpleBot.csproj

# Alternative: Configure environment for full system
cp .env.sample.local .env
# Edit .env with your TopstepX credentials

# For complete system (requires additional setup)
# dotnet run --project src/UnifiedOrchestrator
```

✅ **VERIFIED WORKING**: The bot successfully launches with 0 errors and 0 warnings

## 🔍 **Bot Diagnostics & Monitoring**

### Self-Hosted Bot Launch Diagnostics

For self-hosted runner deployments, use the **Bot Launch Diagnostics** workflow to capture complete startup information and logs:

**GitHub Actions → "🤖 Bot Launch Diagnostics - Self-Hosted"**

**What it captures:**
- ✅ Complete console output from bot startup
- ✅ All error messages and stack traces
- ✅ System and environment information
- ✅ Structured JSON logs with parsed events
- ✅ Runtime metrics and performance data
- ✅ Configuration validation results

**How to use:**
1. Navigate to **Actions** tab in GitHub
2. Select **"🤖 Bot Launch Diagnostics - Self-Hosted"** workflow
3. Click **"Run workflow"**
4. Configure runtime duration (default: 5 minutes)
5. Download artifacts after completion

**Artifacts include:**
- `system-info.json` - System and environment details
- `console-output-*.log` - Complete console logs
- `error-output-*.log` - Error stream output
- `structured-log-*.json` - Parsed startup events with timestamps

**Safety:** Always runs in DRY_RUN mode to prevent live trading during diagnostics.

📚 **Full Documentation**: See [.github/workflows/README-bot-diagnostics.md](.github/workflows/README-bot-diagnostics.md) for detailed usage instructions.

### Safety Configuration
```json
{
  "EnableDryRunMode": true,      // ALWAYS start in dry run
  "EnableAutoExecution": false,  // Require explicit enable
  "MaxDailyLoss": -1000,        // $1000 daily loss limit
  "MaxPositionSize": 5,         // 5 contracts maximum
  "AccountId": "your-account-id"
}
```

## Architecture

### Critical Components
- **EmergencyStopSystem**: Background service monitoring kill.txt
- **PositionTrackingSystem**: Real-time position and risk management
- **OrderFillConfirmationSystem**: Triple verification of all executions
- **ErrorHandlingMonitoringSystem**: Health tracking and alerting
- **TradingSystemIntegrationService**: Unified component coordination

### Event-Driven Safety
- Emergency stop → Cancel all pending orders
- Risk violation → Automatic position limits
- Health degradation → Trading suspension
- Connection loss → Safe shutdown procedures

## Risk Management

### ES/MES Trading Rules
```csharp
// Tick rounding (0.25 precision)
decimal roundedPrice = Math.Round(price / 0.25m, 0) * 0.25m;

// Risk validation
if (risk <= 0) throw new InvalidOperationException("Risk must be > 0");

// R multiple calculation  
decimal rMultiple = (isLong ? target - entry : entry - target) / risk;
```

### Order Flow Requirements
1. **Unique Order ID**: S11L-YYYYMMDD-HHMMSS-{guid}
2. **API Confirmation**: OrderId returned from REST call
3. **SDK Verification**: Trade execution via TopstepX adapter
4. **Position Update**: Real-time P&L calculation

## Monitoring

### Health Dashboard
- System health score (0-100%)
- Component status (Healthy/Warning/Critical)
- Recent error count and severity
- Trading enablement status

### Automated Alerts
- Critical errors → Emergency log files
- Risk violations → Immediate notifications  
- System health degradation → Trading suspension
- Connection issues → Reconnection attempts

## Security

### API Security
- Bearer token authentication
- Environment variable storage (never hardcoded)
- Request rate limiting
- Connection encryption (HTTPS/WSS)

### Audit Compliance
- All orders logged with structured format
- Fill confirmations with timestamps
- Error tracking with unique identifiers
- Health reports every 5 minutes

## Development

### Adding New Strategies
1. Implement in `src/StrategyAgent`
2. Register with `TradingSystemIntegrationService`
3. Add health monitoring integration
4. Test in DRY_RUN mode first

### Error Handling
```csharp
try {
    // Trading logic
} catch (Exception ex) {
    await errorMonitoring.LogErrorAsync("ComponentName", ex, ErrorSeverity.High);
    // Handle gracefully
}
```

### Testing
```bash
# Run system tests
dotnet test tests/

# Health check endpoint
curl https://localhost:5001/health

# Emergency stop test
echo "Emergency Stop Test" > kill.txt
```

## Deployment

### Environment Setup
- Configure TopstepX API credentials
- Set risk limits appropriate for account size  
- Enable logging and monitoring
- Test emergency procedures

### Production Checklist
- [ ] DRY_RUN mode tested thoroughly
- [ ] Emergency stop procedures verified
- [ ] Risk limits configured correctly
- [ ] Health monitoring operational
- [ ] Backup and recovery tested

---

**⚠️ IMPORTANT**: Always start in DRY_RUN mode and verify all systems before enabling live trading. The emergency stop system (kill.txt) should be tested before any live deployment.