# Trading Bot Project Structure

This document describes the clean, organized structure of the trading bot project after comprehensive cleanup.

## Core Application (`/src`)

### Main Components
- **`BotCore/`** - Core trading logic, strategies (S1-S14), market data, API clients
- **`OrchestratorAgent/`** - Main orchestration service, execution logic, health monitoring
- **`StrategyAgent/`** - Strategy execution agent
- **`UpdaterAgent/`** - Automated deployment and update management
- **`TopstepAuthAgent/`** - Authentication service for TopstepX API
- **`StandaloneDashboard/`** - Web dashboard for monitoring (if present)

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

Organized test files moved from root:
- **Integration tests** - `test_*_integration.*`
- **Component tests** - `test_*_demo.*`, `test_*_components.*`
- **System demos** - `demo_24_7_system.cs`

## Configuration & Documentation

### Configuration
- **`appsettings.json`** - Main application configuration
- **`.env.sample.local`** - Environment variables template
- **`TopstepX.Bot.sln`** - Visual Studio solution file

### Documentation
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

```bash
# Restore dependencies
dotnet restore

# Build solution
dotnet build

# Run main orchestrator
dotnet run --project src/OrchestratorAgent

# Run tests
dotnet test  # (when test projects are added to solution)
```

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

The project is now clean, organized, and ready for production deployment!