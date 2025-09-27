# Operations Scripts Directory  

This directory contains operational and deployment scripts that were previously in the root directory.

## Scripts

### Production & Deployment
- `deploy-production.sh` - Production deployment script
- `production-demo.sh` - Production readiness demonstration
- `launch-production.sh` - Production launch script  
- `launch-paper-trading.sh` - Paper trading launch script

### Verification & Validation
- `verify-production-ready.sh` - Production readiness verification
- `verify-core-guardrails.sh` - Core safety guardrails verification  
- `verify-system.sh` - General system verification
- `verify-real-data-integration.sh` - Real data integration verification
- `verify-es-nq-production-only.sh` - ES/NQ production environment verification

## Usage

All scripts should be run from the repository root directory:

```bash
# Example usage
./scripts/operations/production-demo.sh
./scripts/operations/verify-production-ready.sh
```

## Important Notes

- Scripts maintain their original functionality and parameters
- Path references within scripts may need updating as they are consumed
- All production safety mechanisms and guardrails are preserved

## Cleanup Date

Reorganized: $(date)  
Action: Operations script consolidation per requirements