# Archive Directory

This directory contains files and components that have been moved from the root during repository cleanup and reorganization.

## Structure

### `demos/`
- **`full-automation/`** - Complete demo_full_automation directory moved here for historical reference
- **`DemoRunner/`** - Empty placeholder demo project moved from samples/

## Purpose

These archived components are preserved for:
- Historical reference and documentation
- Potential future migration needs
- Safety - avoiding direct deletion of potentially referenced code

## Status

All components in this directory are considered **inactive** and not part of the current operational system. The active system uses:
- **UnifiedOrchestrator** as the main entry point
- **MinimalDemo** for smoke testing (until fully migrated to UnifiedOrchestrator)
- All production code remains in `src/` directory

## Cleanup Date

Reorganized: $(date)
Action: Repository structure cleanup per requirements