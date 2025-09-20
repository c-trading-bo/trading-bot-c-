# Trading Bot Project Structure

This document describes the clean, organized structure of the trading bot project after comprehensive cleanup.

> **🤖 For Coding Agents**: See also `CODING_AGENT_GUIDE.md` for development workflows and `/.github/copilot-instructions.md` for detailed instructions.

## Core Application (`/src`)

> **🤖 Agent Entry Points**: Start with `UnifiedOrchestrator/` for main flow, `BotCore/Services/` for DI setup, and `TopstepAuthAgent/` for API integration.

### Main Components
- **`BotCore/`** - Core trading logic, strategies (S1-S14), market data, API clients
- **`UnifiedOrchestrator/`** - **Primary entry point** - Main application orchestrator 
- **`OrchestratorAgent/`** - Main orchestration service, execution logic, health monitoring
- **`StrategyAgent/`** - Strategy execution agent
- **`UpdaterAgent/`** - Automated deployment and update management
- **`TopstepAuthAgent/`** - **Key for agents** - Authentication service for TopstepX API
- **`Abstractions/`** - Shared interfaces and contracts
- **`Infrastructure/`** - Infrastructure services (alerts, monitoring)
  - **`Alerts/`** - Email/Slack alert system
- **`ML/`** - Machine learning components
  - **`HistoricalTrainer/`** - Historical data training
- **`RLAgent/`** - Reinforcement learning agent
- **`Strategies/`** - Trading strategy implementations
- **`IntelligenceAgent/`** - AI/ML intelligence processing
- **`IntelligenceStack/`** - Intelligence processing stack
- **`Monitoring/`** - System monitoring and observability
- **`Safety/`** - **Critical** - Safety and risk management controls

## Intelligence System (`/Intelligence`)

Complete AI/ML pipeline running separately from main trading bot:
- **`scripts/`** - Data collection, model training, signal generation
- **`data/`** - Raw data, features, signals, trade results
- **`models/`** - Trained ML models
- **`reports/`** - Daily analysis reports

## Organization Scripts (`/scripts`)

### By Category
- **`monitoring/`** - System health and performance monitoring scripts
- **`validation/`** - Validation and verification scripts
- **`utilities/`** - General utility and maintenance scripts
- **`powershell/`** - PowerShell scripts for Windows environments
- **`windows/`** - Windows-specific utilities

### Key Scripts
- **`cleanup-project-simple.ps1`** - Project cleanup automation
- **`cloud-ml-training.sh`** - Cloud ML training setup
- **`run-bot-persist.ps1`** - Persistent bot execution

## Testing (`/tests`)

> **🤖 Agent Testing**: Use `dotnet test tests/Unit/MLRLAuditTests.csproj` for focused testing. Build warnings are expected.

Organized test files moved from root:
- **`Unit/`** - Unit tests including `MLRLAuditTests/`
- **`Integration/`** - Integration tests
- **`SimpleBot.Tests/`** - Simple bot component tests
- **`mocks/`** - Test mocks and fixtures
- **Integration tests** - `test_*_integration.*`
- **Component tests** - `test_*_demo.*`, `test_*_components.*`
- **System demos** - `demo_24_7_system.cs`

## Configuration & Documentation

> **🤖 Agent Config**: Main config in `src/BotCore/Services/ProductionConfigurationService.cs`, environment in `.env`, build rules in `Directory.Build.props`.

### Configuration
- **`appsettings.json`** - Main application configuration
- **`.env`** - **Key for agents** - Environment variables (copy from `.env.example`)
- **`Directory.Build.props`** - **Important** - Global build configuration and analyzer rules
- **`TopstepX.Bot.sln`** - Visual Studio solution file

### Documentation
- **`.github/copilot-instructions.md`** - **Essential for agents** - Comprehensive Copilot instructions
- **`CODING_AGENT_GUIDE.md`** - **Essential for agents** - Quick start guide for coding agents
- **`ENHANCED_DATA_COLLECTION_ARCHITECTURE.md`** - Data collection system
- **`PRODUCTION_DEPLOYMENT_GUIDE.md`** - Deployment instructions
- **`GITHUB_SECRETS_SETUP.md`** - GitHub Actions setup
- **`LOCAL_HTTPS_DASHBOARD.md`** - Dashboard setup guide
- **`docs/`** - Detailed technical documentation

## Data & State

- **`data/`** - Application data storage
- **`state/`** - Runtime state and configuration
- **`models/`** - ML models and trained components
- **`reports/`** - Generated reports and analytics

## GitHub Actions (`.github/workflows`)

Automated CI/CD, data collection, and ML training workflows.

## Removed During Cleanup

### Empty/Placeholder Files Removed
- Empty C# files: `CloudRlTrainerV2.cs`, `SmartExecutionModel.cs`, etc.
- Empty Python ML training scripts: `train_*.py` placeholders
- Empty test projects: `OnnxTestRunner`, `DemoRunner`, `TestEnhancedZones`, `RLComponentTest`
- Empty documentation: `RUNBOOK.md`, `Cloud-Local-Trading-Architecture.md`

### Temporary Files Removed
- `temp_backup/`, `temp_extract/`, `temp_models/` directories
- `workflow_*.json`, `test_results.json` temporary result files
- `*FIX*COMPLETE*.md`, `ULTIMATE_*.md` status documentation

### Redundant Scripts Removed
- `fix_all_workflows.sh`, `ultimate_fix.sh` - temporary fix scripts
- `tools/sign_models.py` - empty utility file

## Build & Run

> **🤖 Agent Commands**: Always start with `dotnet restore`, expect build warnings, use `UnifiedOrchestrator` as main entry point.

```bash
# Restore dependencies (always do this first)
dotnet restore

# Build solution (expect analyzer warnings - don't fix unless asked)
dotnet build --no-restore

# Run main trading bot application
dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj

# Run tests (build warnings expected)
dotnet test --no-build --verbosity normal

# Run specific test project
dotnet test tests/Unit/MLRLAuditTests.csproj
```

## Single Entry Point

> **🤖 Agent Note**: The main entry point is now `UnifiedOrchestrator`. Build warnings from analyzers are expected and should not be fixed unless specifically requested.

✅ **Primary Entry Point**: `src/UnifiedOrchestrator/UnifiedOrchestrator.csproj`
- Command: `dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj`
- Status: ✅ Core functionality intact (analyzer warnings expected)
- Features: Unified orchestration, trading logic, ML/RL integration

📝 **Alternative Entry Point**: `SimpleBot/SimpleBot.csproj` (legacy)
- Command: `dotnet run --project SimpleBot/SimpleBot.csproj`
- Status: ✅ Builds and runs with 0 errors and 0 warnings
- Features: Core strategy system validation and health checks

## Key Features Preserved

✅ **All core trading functionality intact**  
✅ **Complete Intelligence/ML pipeline**  
✅ **Comprehensive monitoring and health checks**  
✅ **Automated deployment and updates**  
✅ **GitHub Actions workflows**  
✅ **All strategies (S1-S14) preserved**  
✅ **TopstepX API integration**  
✅ **Dashboard and monitoring tools**  

## Next Steps

1. **Production deployment** using guides in `/docs`
2. **Testing** with organized test files in `/tests`
3. **Monitoring** using scripts in `/scripts/monitoring`
4. **Intelligence integration** following `/Intelligence/README.md`

## ✅ CURRENT STATUS: OPERATIONAL

The project is now clean, organized, and ready for production deployment!

> **🤖 For Coding Agents**: Build warnings from static analyzers are expected. Focus on functional changes, not code quality fixes unless specifically requested.

**🎯 VERIFIED WORKING**: The trading bot successfully launches with core functionality intact:
```bash
# Main application (analyzer warnings expected)
dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj

# Legacy entry point (0 errors, 0 warnings)
dotnet run --project SimpleBot/SimpleBot.csproj
```

### 🤖 Coding Agent Quick Reference
- **Start here**: `CODING_AGENT_GUIDE.md` and `.github/copilot-instructions.md`
- **Main entry**: `src/UnifiedOrchestrator/`
- **Core services**: `src/BotCore/Services/`
- **API integration**: `src/TopstepAuthAgent/`
- **Configuration**: `.env` (copy from `.env.example`)
- **Build command**: `dotnet restore && dotnet build --no-restore`
- **Test command**: `dotnet test --no-build --verbosity normal`