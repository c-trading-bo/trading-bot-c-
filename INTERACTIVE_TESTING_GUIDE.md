# Interactive Testing Guide

## Overview

This guide explains how to test your trading bot functions and logic in a **real environment** (not a sandbox) without executing live trades. This is designed to help when code agents struggle to debug bot logic, allowing you to step through trading decisions interactively with real market data.

## Why Interactive Testing?

### Problems This Solves

1. **Code agents struggle with debugging** - Automated tools can't always understand complex trading logic
2. **Sandbox environments are too isolated** - Mocked data doesn't reveal real-world issues
3. **Need to inspect internal state** - See what the bot is actually doing step-by-step
4. **Want to test specific functions** - Validate individual components without full system runs

### Safety First

✅ **All interactive testing runs in DRY_RUN mode** - No live trades are ever executed  
✅ **Production guardrails enforced** - Kill switch, risk validation, order evidence still active  
✅ **Real data feeds** - Connect to actual market data for realistic testing  
✅ **Step-by-step control** - You decide when each step executes  

## Quick Start

### Method 1: Interactive Mode (Step-by-Step Debugging)

Run the bot in interactive mode to step through trading logic:

```bash
./dev-helper.sh run-interactive
```

Or directly:

```bash
dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj -- --interactive
```

**What you get:**
- Step-by-step execution of trading decisions
- Real market data feeds (no mocking)
- Ability to inspect brain outputs and internal state
- Commands to pause, continue, or test specific functions
- All trades are DRY_RUN (no live execution)

**Interactive Commands:**
- `[Enter]` - Execute next step
- `c` - Continue (disable step mode)
- `p` - Pause execution
- `s` - Show current state
- `i` - Inspect brain output
- `b` - Test specific strategy/function
- `q` - Quit interactive mode

### Method 2: Test Specific Functions

Test individual bot functions in isolation:

```bash
# Test risk calculation logic
./dev-helper.sh test-function risk-calc

# Test ES/MES price rounding (0.25 tick size)
./dev-helper.sh test-function tick-round

# Test order evidence validation
./dev-helper.sh test-function order-proof

# Test a specific strategy
./dev-helper.sh test-function strategy S6

# Test market data parsing
./dev-helper.sh test-function market-data
```

Or directly:

```bash
dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj -- --test-function risk-calc
```

## Available Test Functions

### 1. Risk Calculation (`risk-calc`)

Tests the risk calculation logic that validates trades.

```bash
./dev-helper.sh test-function risk-calc
```

**What it tests:**
- Calculates risk (entry - stop)
- Calculates reward (target - entry)
- Computes R-multiple (reward / risk)
- Validates that risk > 0 (rejects invalid trades)

**Example output:**
```
Standard 2R trade:
  Entry:  $4500.00
  Stop:   $4495.00
  Target: $4510.00
  Risk:   $5.00 (✅)
  Reward: $10.00
  R-Multiple: 2.00x
  Valid:  ✅ PASS
```

### 2. Tick Rounding (`tick-round`)

Tests ES/MES price rounding to 0.25 tick increments.

```bash
./dev-helper.sh test-function tick-round
```

**What it tests:**
- Rounds prices to nearest 0.25 tick
- Validates ES/MES tick size compliance
- Shows before/after rounding

**Example output:**
```
$4500.13 → $4500.00 ⚙️
$4500.25 → $4500.25 ✅
$4500.38 → $4500.50 ⚙️
$4500.63 → $4500.50 ⚙️
```

### 3. Order Evidence (`order-proof`)

Tests order evidence validation requirements.

```bash
./dev-helper.sh test-function order-proof
```

**What it tests:**
- Validates orderId is present
- Validates fill event is present
- Requires BOTH for valid order
- Rejects orders without full evidence

**Example output:**
```
Valid order with evidence:
  OrderId:    ABC123 ✅
  FillEvent:  FILL_001 ✅
  Valid:      ✅ ACCEPT

OrderId without fill event:
  OrderId:    ABC124 ✅
  FillEvent:  (none) ❌
  Valid:      ❌ REJECT - Need both OrderId AND FillEvent
```

### 4. Strategy Testing (`strategy`)

Tests a specific strategy configuration.

```bash
./dev-helper.sh test-function strategy S6
```

**What it tests:**
- Loads strategy configuration from `config/strategy.{name}.json`
- Validates configuration format
- Shows strategy parameters
- Framework for strategy-specific logic tests

### 5. Market Data Parsing (`market-data`)

Tests market data parsing and validation.

```bash
./dev-helper.sh test-function market-data
```

**What it tests:**
- Parses sample market data
- Validates data format
- Shows parsed fields (symbol, price, volume, time)

## Interactive Mode Workflow

### Starting Interactive Mode

```bash
./dev-helper.sh run-interactive
```

You'll see:

```
🧪 INTERACTIVE TESTING MODE
================================================================================
Debug your bot logic step-by-step with real market data

✅ Safety: DRY_RUN mode enforced - no live trades
✅ Real Data: Connected to actual market data feeds  
✅ Step Mode: Execute trading logic interactively
✅ Inspect: View internal state and brain outputs
✅ Test: Run specific strategies or functions in isolation
================================================================================

Commands:
  [Enter]    - Execute next step
  'c'        - Continue (disable step mode)
  'p'        - Pause execution
  's'        - Show current state
  'i'        - Inspect brain output
  'b'        - Test specific strategy/function
  'q'        - Quit interactive mode
================================================================================
```

### Step-by-Step Execution

1. **Press Enter** to execute the next trading decision step
2. The bot will:
   - Fetch real market data
   - Run brain decision logic
   - Log the decision (no actual trade)
   - Wait for your next command

3. **Type 's'** to see current state:
```
📊 CURRENT STATE
================================================================================
Step Counter: 5
Step Mode: true
Paused: false
Brain Available: true
Market Data Provider Available: true
================================================================================
```

4. **Type 'i'** to inspect brain output:
```
🧠 BRAIN OUTPUT INSPECTION
================================================================================
✅ Brain instance available
Brain type: UnifiedTradingBrain
[Detailed brain state information]
================================================================================
```

5. **Type 'b'** to test a specific function interactively

### Continuous Mode

Type `c` to disable step mode and run continuously:
```
▶️ Continuous mode enabled
```

The bot will execute trading logic automatically with a 5-second delay between steps.

### Pausing

Type `p` to pause/resume:
```
⏸️ Paused
```

Press `p` again to resume.

## Integration with Existing Tests

The interactive testing mode complements existing test infrastructure:

### Unit Tests
- Location: `tests/Unit/`
- Purpose: Test individual components in isolation
- Use when: Testing pure logic without external dependencies

### Integration Tests
- Location: `tests/Integration/`
- Purpose: Test component interactions
- Use when: Validating service integration

### Interactive Testing
- Location: `dev-helper.sh run-interactive`
- Purpose: Debug with real data in controlled environment
- Use when: Code agents struggle or you need to inspect runtime behavior

### Production Testing
- Location: `dev-helper.sh run` (with DRY_RUN=1)
- Purpose: Full system test with real data
- Use when: Final validation before live trading

## Example Workflows

### Debugging a Strategy Issue

1. Start interactive mode:
   ```bash
   ./dev-helper.sh run-interactive
   ```

2. Step through trading decisions (press Enter for each step)

3. When you see unexpected behavior, type `s` to check state

4. Type `i` to inspect brain output

5. Type `b` to test the specific strategy in isolation

6. Fix the issue in code

7. Restart interactive mode to verify fix

### Validating Risk Calculation

1. Test the risk calculation function:
   ```bash
   ./dev-helper.sh test-function risk-calc
   ```

2. Review the test cases and results

3. Add your own test cases by modifying the function in `Program.cs`

4. Re-run to verify

### Testing a New Strategy

1. Create strategy config in `config/strategy.{name}.json`

2. Test the strategy:
   ```bash
   ./dev-helper.sh test-function strategy {name}
   ```

3. Verify configuration loads correctly

4. Add strategy-specific test logic

5. Run in interactive mode to see strategy in action:
   ```bash
   ./dev-helper.sh run-interactive
   ```

## Troubleshooting

### Interactive mode not responding

**Issue:** Bot seems stuck in interactive mode  
**Solution:** Press Enter to execute next step, or type 'c' for continuous mode

### Can't test specific function

**Issue:** `./dev-helper.sh test-function` shows error  
**Solution:** Ensure you provide a function name:
```bash
./dev-helper.sh test-function risk-calc
```

### Want to add custom test function

**Issue:** Need to test logic not covered by existing functions  
**Solution:** 
1. Add your test function to `Program.cs` in the `RunTestFunctionAsync` method
2. Add a case to the switch statement
3. Implement your test logic
4. Update `dev-helper.sh` help text

### DRY_RUN not enforced

**Issue:** Worried about live trades  
**Solution:** Interactive and test modes ALWAYS force DRY_RUN:
```csharp
Environment.SetEnvironmentVariable("DRY_RUN", "true");
Environment.SetEnvironmentVariable("TRADING_MODE", "DRY_RUN");
```

## Safety Guarantees

All interactive testing modes enforce these safety guarantees:

1. **DRY_RUN Mode**: Hardcoded - cannot be disabled in interactive mode
2. **No Live Trades**: Order execution is always simulated
3. **Kill Switch Active**: `kill.txt` monitoring still functional
4. **Risk Validation**: Trades with risk ≤ 0 still rejected
5. **Order Evidence**: OrderId + FillEvent still required for fills
6. **Tick Rounding**: ES/MES 0.25 tick rounding still enforced

## Command Reference

### dev-helper.sh Commands

```bash
# Interactive testing
./dev-helper.sh run-interactive

# Test specific functions
./dev-helper.sh test-function risk-calc
./dev-helper.sh test-function tick-round
./dev-helper.sh test-function order-proof
./dev-helper.sh test-function strategy S6
./dev-helper.sh test-function market-data

# Traditional testing
./dev-helper.sh test          # Run all tests
./dev-helper.sh test-unit     # Run unit tests only
./dev-helper.sh run-smoke     # Run smoke test
```

### Direct dotnet Commands

```bash
# Interactive mode
dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj -- --interactive

# Test function
dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj -- --test-function risk-calc

# With arguments
dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj -- --test-function strategy S6
```

## Summary

Interactive testing provides a way to:

✅ **Debug bot logic** when code agents struggle  
✅ **Test with real data** while maintaining safety (DRY_RUN)  
✅ **Step through decisions** to understand what the bot is doing  
✅ **Inspect internal state** to diagnose issues  
✅ **Validate specific functions** in isolation  
✅ **Develop with confidence** knowing production guardrails are active  

This bridges the gap between isolated unit tests and full production runs, giving you the control and visibility needed to develop and debug trading bot logic effectively.
