# Phase 0: Automated Discovery & Audit Tools

This directory contains automated tools for Phase 0 of the Training Mode Split architecture project.

## Purpose

These tools analyze the existing codebase **without making any changes** to identify:
- All training systems and their complexity
- Data flow through the system  
- Configuration requirements for mode separation
- Safety-critical code that must stay in live mode

## Quick Start

Run the complete audit with one command:

```bash
python3 tools/audit/run_phase0_audit.py
```

This will execute all 4 audit steps and generate a comprehensive report.

## Individual Scripts

You can also run individual audit steps:

### Step 1: Training Systems Discovery
```bash
python3 tools/audit/discover_training_systems.py
```

**Output:** `reports/training_systems_audit.json`

Identifies all learning systems in the codebase and classifies them by complexity:
- **HEAVY**: Multi-epoch training, gradient descent (→ needs Sunday training window)
- **MEDIUM**: Statistical updates, retraining (→ could fit in 15-min daily window)
- **LIGHT**: Online learning, immediate feedback (→ stays in live mode)

### Step 2: Data Flow Tracing
```bash
python3 tools/audit/trace_data_flow.py
```

**Output:** `reports/data_flow_analysis.json`

Traces how data flows through the system:
- Experience creation points
- Storage operations
- Loading mechanisms
- Training/processing points

Includes a Mermaid diagram for visualization.

### Step 3: Configuration Validation
```bash
python3 tools/audit/validate_configuration.py
```

**Output:** `reports/configuration_validation.json`

Validates the configuration is ready for mode separation:
- Checks .env file and required variables
- Validates mode settings (no conflicts)
- Verifies required directories exist
- Confirms futures market hours configuration

### Step 4: Safety Systems Inventory
```bash
python3 tools/audit/inventory_safety_systems.py
```

**Output:** `reports/safety_systems_inventory.json`

Inventories all safety-critical code:
- TopStep enforcement
- Broker connections
- Order execution
- Risk management
- Position management

All safety systems must stay in Live Mode (Historical Mode operates offline).

## Reports Directory

All reports are saved to `reports/` directory:
- `training_systems_audit.json` - Training complexity classification
- `data_flow_analysis.json` - Data flow diagram and analysis
- `configuration_validation.json` - Config validation results
- `safety_systems_inventory.json` - Safety components inventory
- `phase0_master_audit.json` - Master report combining all audits

## Requirements

- Python 3.7+
- No external dependencies (uses only standard library)

## Architecture Context

This audit supports the **Futures-Aware Two-Mode Architecture**:

**Live Mode** (23 hours/day, Mon-Fri):
- Fast trading decisions (<10ms)
- No heavy training
- All safety systems active
- Real broker connections

**Historical Mode** (Sunday 12 PM - 5:45 PM):
- Heavy training (5+ hours)
- Offline (no broker connections)
- Replays 90-day historical data
- Trains all models

**Daily Mini-Training** (5:00 PM - 5:15 PM):
- 15-minute maintenance window
- Quick parameter updates only
- Hot-swap model updates

## Next Steps

After reviewing the audit reports:

1. **Review Findings**: Check all JSON reports in `reports/` directory
2. **Stakeholder Discussion**: Review timeline and effort estimates
3. **Architectural Decisions**: Use audit data to inform split approach
4. **Phase 1**: Begin architecture design if proceeding with split

## Important Notes

⚠️ **This is audit-only** - No code is modified by these scripts

✅ **Safe to run** - Only reads files and generates reports

📊 **Data-driven** - Provides objective metrics for decision-making

🎯 **Architectural planning** - Informs the training mode split design

## Support

For questions about the audit results or the training split architecture, refer to:
- `TRAINING_SPLIT_PLAN.md` - Complete architecture plan
- `ML_LEARNING_COMPONENTS_AUDIT.md` - Previous manual audit
- `HISTORICAL_MODE_IMPLEMENTATION_SUMMARY.md` - Current historical mode

---

**Status**: Phase 0 - Automated discovery and audit tools  
**Next Phase**: Phase 1 - Architecture design and implementation planning
