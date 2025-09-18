# Configuration Migration Guide - Legacy to SDK

This guide helps migrate from legacy configuration to the new SDK-based configuration system.

## Deprecated Environment Variables

The following environment variables have been deprecated and should be removed:

- `TOPSTEPX_LEGACY_CLIENT_ID` → Use `PROJECT_X_API_KEY`
- `TOPSTEPX_LEGACY_SECRET` → Use `PROJECT_X_USERNAME` 
- `INFRASTRUCTURE_TOPSTEPX_ENABLED` → Always enabled in SDK mode
- `LEGACY_SIMULATION_MODE` → Automatic simulation when SDK unavailable
- `OLD_HISTORICAL_DATA_PATH` → Uses SDK adapter historical data
- `LEGACY_RISK_MANAGER_ENABLED` → Always enabled with centralized risk defaults

## New Configuration Structure

### Risk Management
Risk defaults are now centralized in `src/Safety/RiskDefaults.cs`:
- Environment-specific risk limits (Production, Staging, Development)
- SDK adapter specific risk defaults
- ML/RL specific risk thresholds
- UCB algorithm risk parameters

### SDK Configuration
Required environment variables:
```bash
PROJECT_X_API_KEY=your_api_key
PROJECT_X_USERNAME=your_username
TRADING_ENVIRONMENT=DEVELOPMENT|STAGING|PRODUCTION
```

### Optional Overrides
```bash
MAX_DAILY_LOSS_OVERRIDE=1000
MAX_POSITION_SIZE_OVERRIDE=10000
ML_MODEL_UPDATE_INTERVAL_HOURS=24
UCB_CONFIDENCE_THRESHOLD=0.65
```

## Migration Steps

1. Remove all legacy environment variables from your `.env` files
2. Add the new SDK environment variables
3. Update any code that references old config keys
4. Use `RiskDefaults.GetConfigurationForEnvironment()` for risk settings
5. Test the system with the new configuration

## Validation

Run the following to validate your configuration:
```bash
python3 python/test_sdk_wiring.py
python3 runtime_proof_demo.py
```

Both should pass without errors and show "SDK-only" status.
