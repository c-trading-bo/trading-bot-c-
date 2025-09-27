# Legacy Projects Directory

This directory contains legacy projects that are no longer part of the main TopstepX.Bot.sln solution but are preserved for reference.

## Moved Projects

### `TradingBot.Orchestrators/`
- Legacy orchestrator implementation
- Not referenced in current solution
- Replaced by UnifiedOrchestrator

### `TradingBot/`
- Legacy trading bot application from app/ directory  
- Not referenced in current solution
- Replaced by UnifiedOrchestrator

## Important Notes

- **DO NOT** attempt to build or run these projects as part of the main solution
- These are preserved for historical reference and potential migration of specific features
- The Makefile has been updated to reference UnifiedOrchestrator instead of these legacy components

## Migration Status

The functionality from these legacy projects has been consolidated into:
- `src/UnifiedOrchestrator/` - Main application entry point
- `src/BotCore/` - Core trading logic and services
- `src/Strategies/` - Trading strategies
- Other modular components in `src/`

## Cleanup Date

Reorganized: $(date)
Action: Legacy project consolidation per requirements