# 🤖 Coding Agent Quick Start

Welcome, coding agent! This repository is set up for optimal agent development experience.

## 🛡️ IMPORTANT: Follow Production Guardrails

**Before making any changes, read `.github/copilot-instructions.md` for complete production guardrails.**

❌ Never modify config files, add suppressions, or bypass safety mechanisms  
✅ Always run `./dev-helper.sh analyzer-check` before committing  

## ⚡ 30-Second Setup

```bash
# 1. Setup environment
./dev-helper.sh setup

# 2. Build solution (warnings expected)
./dev-helper.sh build

# 3. Ready to code!
```

## 📚 Essential Reading (in order)

1. **`.github/copilot-instructions.md`** - **MUST READ** - Production guardrails and safety requirements
2. **This file** - Quick orientation
3. **`CODING_AGENT_GUIDE.md`** - Comprehensive development guide
4. **`PROJECT_STRUCTURE.md`** - Repository layout

## 🎯 Key Entry Points

| Task | File/Directory | Command |
|------|---------------|---------|
| **Main application** | `src/UnifiedOrchestrator/` | `./dev-helper.sh run` |
| **API integration** | `src/TopstepAuthAgent/` | Edit API clients here |
| **Core services** | `src/BotCore/Services/` | DI container setup |
| **Trading strategies** | `src/Strategies/` | Strategy implementations |
| **Configuration** | `.env` | Copy from `.env.example` |
| **Tests** | `tests/Unit/` | `./dev-helper.sh test` |

## ⚠️ Important Rules

### ✅ Do
- Use `decimal` for all monetary values
- Follow async/await patterns with `ConfigureAwait(false)`
- Round ES/MES prices to 0.25 increments using `Px.RoundToTick()`
- Require order proof (orderId + fill confirmation)
- Make minimal, surgical changes only

### ❌ Don't
- Fix analyzer warnings unless specifically asked
- Touch production secrets (`.env`, `kill.txt`)
- Bypass safety mechanisms (`DRY_RUN`, risk limits)
- Log sensitive data (tokens, passwords)
- Make large architectural changes

## 🔧 Development Commands

```bash
# Quick development cycle
./dev-helper.sh setup    # Setup environment
./dev-helper.sh build    # Build solution

# Validation workflow (complete agent workflow)
./dev-helper.sh analyzer-check  # Ensure no new warnings
./dev-helper.sh backtest        # Validate with local sample data
./dev-helper.sh riskcheck       # Check risk constants

# Testing
./dev-helper.sh test     # Run tests

# Run applications
./dev-helper.sh run          # Main app (UnifiedOrchestrator)
./dev-helper.sh run-simple   # Legacy SimpleBot (clean build)

# Testing
./dev-helper.sh test-unit    # Unit tests only
dotnet test tests/Unit/MLRLAuditTests.csproj  # Specific test project

# Utilities
./dev-helper.sh clean    # Clean build artifacts
./dev-helper.sh full     # Full cycle: setup -> build -> test
```

## 🚨 Expected Build Behavior

- **Analyzer warnings are normal** - Don't fix unless asked
- **Build may show ~1500+ warnings** - This is expected due to strict analyzer rules
- **Focus on functionality** - Not code quality unless specifically requested
- **Tests may have some failures** - Legacy/integration issues, not your responsibility

## 📁 Quick Navigation

```
src/
├── UnifiedOrchestrator/    # 🎯 Main entry point
├── BotCore/Services/       # 🔧 Core services & DI
├── TopstepAuthAgent/       # 🔌 API integration
├── Strategies/             # 📈 Trading strategies
├── RLAgent/               # 🧠 ML/RL components
└── Safety/                # 🚨 Safety systems (be careful!)

.github/
└── copilot-instructions.md # 📖 Comprehensive context

tests/
└── Unit/MLRLAuditTests/   # 🧪 Main test project
```

## 🆘 Common Issues

| Problem | Solution |
|---------|----------|
| Build fails | `dotnet restore` then `dotnet build --no-restore` |
| Missing .env | Copy from `.env.example` |
| Port 5050 in use | Check for other processes or use different port |
| Test failures | Expected for integration tests, focus on unit tests |
| Too many warnings | Normal! Don't fix unless asked |

## 💡 Pro Tips

1. **Start with existing patterns** - Don't reinvent, follow what's there
2. **Check git history** - See how similar changes were made
3. **Use the helper script** - `./dev-helper.sh` for common tasks
4. **Read the context** - Copilot instructions have detailed API docs
5. **Test early and often** - Build and test after small changes

## 🔗 Links

- **Development Guide**: `CODING_AGENT_GUIDE.md`
- **Project Structure**: `PROJECT_STRUCTURE.md`
- **Copilot Instructions**: `.github/copilot-instructions.md`
- **Helper Script**: `./dev-helper.sh help`

---

**Ready to code? Start with**: `./dev-helper.sh setup && ./dev-helper.sh build`

*This README is specifically optimized for coding agent onboarding and development.*