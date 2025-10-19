# Interactive Testing Quick Reference

## When Code Agents Struggle to Debug

Use interactive testing to debug bot logic in a real environment (not a sandbox) with full safety (DRY_RUN enforced).

## Quick Commands

### Interactive Mode (Step-by-Step Debugging)

```bash
# Start interactive mode
./dev-helper.sh run-interactive

# Commands during interactive session:
#   [Enter]  - Execute next step
#   c        - Continue (disable step mode)
#   p        - Pause/resume
#   s        - Show current state
#   i        - Inspect brain output
#   b        - Test specific function
#   q        - Quit
```

### Test Specific Functions

```bash
# Risk calculation (validates risk > 0)
./dev-helper.sh test-function risk-calc

# ES/MES tick rounding (0.25 increments)
./dev-helper.sh test-function tick-round

# Order evidence (requires orderId + fillEvent)
./dev-helper.sh test-function order-proof

# Strategy testing (S2, S3, S6, S11)
./dev-helper.sh test-function strategy S6

# Market data parsing
./dev-helper.sh test-function market-data
```

## Safety Guarantees

✅ **DRY_RUN Mode**: Always enforced - no live trades  
✅ **Production Guardrails**: Kill switch, risk validation, order evidence active  
✅ **Real Data**: Connects to actual market data feeds  
✅ **Step Control**: You decide when each step executes  

## What You Get

- **Real Environment**: Test with actual market data, not mocked
- **No Live Trades**: DRY_RUN enforced - zero risk
- **Step-by-Step**: Execute trading logic incrementally
- **Inspect State**: View brain outputs and internal state
- **Isolate Functions**: Test specific components independently
- **Debug Effectively**: See exactly what bot is doing

## Example Workflow

### Debugging a Strategy Issue

1. **Start interactive mode**
   ```bash
   ./dev-helper.sh run-interactive
   ```

2. **Step through** - Press Enter for each trading decision

3. **See issue** - Type `s` to check state, `i` to inspect brain

4. **Test in isolation** - Type `b` to test specific strategy

5. **Fix code** - Update the problematic logic

6. **Verify** - Restart interactive mode to confirm fix

### Validating Production Guardrails

```bash
# Test risk calculation (rejects risk ≤ 0)
./dev-helper.sh test-function risk-calc

# Test price rounding (ES/MES 0.25 tick)
./dev-helper.sh test-function tick-round

# Test order evidence (requires both orderId and fillEvent)
./dev-helper.sh test-function order-proof
```

## Documentation

- **Full Guide**: [INTERACTIVE_TESTING_GUIDE.md](INTERACTIVE_TESTING_GUIDE.md)
- **Testing Overview**: [TESTING_GUIDE.md](TESTING_GUIDE.md)
- **Main README**: [README.md](README.md)

## Validation

```bash
# Run all interactive feature tests
./test-interactive-features.sh
```

## Why This Helps

**Problem**: Code agents struggle to debug trading bot logic  
**Solution**: Interactive mode lets you step through decisions with real data  
**Benefit**: See exactly what's happening without live trade risk  

**Problem**: Sandbox environments too isolated  
**Solution**: Connect to real market data feeds  
**Benefit**: Realistic testing reveals real-world issues  

**Problem**: Need to test specific functions  
**Solution**: Test functions in isolation  
**Benefit**: Validate components independently  

## Integration with Existing Tests

- **Unit Tests** (`tests/Unit/`) - Test pure logic
- **Integration Tests** (`tests/Integration/`) - Test component interactions
- **Interactive Testing** (`./dev-helper.sh run-interactive`) - Debug with real data
- **Production** (`./dev-helper.sh run`) - Full system with DRY_RUN

Each serves a different purpose - use the right tool for your debugging needs.
