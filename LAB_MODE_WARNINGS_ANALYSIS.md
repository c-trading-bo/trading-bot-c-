# Lab Mode Warnings - Cannot Be Suppressed

This document catalogs all warnings that appear during Lab Mode training that **CANNOT and SHOULD NOT be suppressed** as they represent expected behavior.

## Category 1: TopstepX Adapter Warnings (Expected in Lab Mode)

### Warning: "TopstepX adapter not connected - cannot get live prices"
- **Frequency**: 66 occurrences per training session
- **Severity**: INFO (logged as WARNING but not an error)
- **Root Cause**: Lab Mode intentionally does NOT connect to live TopstepX API
- **Expected Behavior**: YES - Lab Mode uses pre-loaded JSON historical data files
- **Can Be Fixed**: NO - This is correct behavior
- **Should Be Suppressed**: NO - Provides visibility that we're in offline mode
- **Recommendation**: Add LOG LEVEL filter to reduce verbosity in lab mode

### Warning: "TopstepXAdapter: Connected: False, Health: 0.0%"
- **Frequency**: 2 occurrences per session
- **Severity**: INFO
- **Root Cause**: Health check detects adapter is intentionally disconnected
- **Expected Behavior**: YES - Lab Mode does not require live connection
- **Can Be Fixed**: NO
- **Should Be Suppressed**: NO - Health monitoring should remain active
- **Recommendation**: Update health check to expect disconnected state in LAB_MODE

### Warning: "SYSTEM HEALTH DEGRADED: 3/5 connections healthy (60%)"
- **Frequency**: 2 occurrences per session
- **Severity**: WARNING
- **Root Cause**: TopstepX adapter + Historical processing not active (intentional)
- **Expected Behavior**: YES - Lab Mode has reduced connection requirements
- **Can Be Fixed**: NO
- **Should Be Suppressed**: NO - Overall system health should be monitored
- **Recommendation**: Adjust health threshold in LAB_MODE to 60% as acceptable

## Category 2: First Run / Cold Start Warnings

### Warning: "No experiences found - this may be first training session"
- **Frequency**: 3 occurrences (CVaR-PPO, Slippage, Ensemble)
- **Severity**: WARNING
- **Root Cause**: No prior trading activity to learn from
- **Expected Behavior**: YES - First training session
- **Can Be Fixed**: YES - After bot runs for 1-2 weeks
- **Should Be Suppressed**: NO - Important to know why training was skipped
- **Recommendation**: None - warning is informative

### Warning: "Insufficient experiences for training: 0 < 256"
- **Frequency**: 1 occurrence (CVaR-PPO)
- **Severity**: WARNING
- **Root Cause**: Needs minimum 256 trading experiences for PPO training
- **Expected Behavior**: YES - First run
- **Can Be Fixed**: YES - After accumulating trading experiences
- **Should Be Suppressed**: NO - Explains why component skipped
- **Recommendation**: None - warning helps understand training status

### Warning: "Insufficient experiences for slippage training: 0 < 100"
- **Frequency**: 1 occurrence
- **Severity**: WARNING
- **Root Cause**: Needs minimum 100 order executions for slippage modeling
- **Expected Behavior**: YES - First run
- **Can Be Fixed**: YES - After trading activity
- **Should Be Suppressed**: NO
- **Recommendation**: None

### Warning: "Model file not found: models/rl_model.onnx"
- **Frequency**: 2 occurrences
- **Severity**: WARNING
- **Root Cause**: No previously trained models exist
- **Expected Behavior**: YES - First training session
- **Can Be Fixed**: YES - After successful training with sufficient data
- **Should Be Suppressed**: NO - Informs that models will be created
- **Recommendation**: None

## Category 3: Neural UCB Python Training Warnings

### Warning: "No training data available (no strategy decisions in past 7 days)"
- **Frequency**: 1 occurrence
- **Severity**: WARNING (from Python script)
- **Root Cause**: No strategy selection decisions have been made yet
- **Expected Behavior**: YES - First run, no trading history
- **Can Be Fixed**: YES - After bot makes strategy decisions
- **Should Be Suppressed**: NO - Explains why Neural UCB training was skipped
- **Recommendation**: None - informative message

## Category 4: Historical Data Processing

### Warning: "No historical data processing detected yet"
- **Frequency**: 2 occurrences
- **Severity**: INFO
- **Root Cause**: Historical processing happens during training, not continuously
- **Expected Behavior**: YES - Lab Mode processes data during training session
- **Can Be Fixed**: NO - This is correct behavior
- **Should Be Suppressed**: NO - Health monitoring useful
- **Recommendation**: Update health check to recognize Lab Mode training schedule

## Category 5: Validation Phase Failures

### Error: "Canary tests failed - models unstable or too slow"
- **Frequency**: 1 occurrence
- **Severity**: ERROR (expected)
- **Root Cause**: No models available for canary testing (first run)
- **Expected Behavior**: YES - Cannot test models that don't exist yet
- **Can Be Fixed**: YES - After models are trained with sufficient data
- **Should Be Suppressed**: NO - Important validation step
- **Recommendation**: None - will pass once models are trained

### Error: "Cannot promote - validation failed"
- **Frequency**: 1 occurrence
- **Severity**: ERROR (expected)
- **Root Cause**: Canary tests failed (expected - no models)
- **Expected Behavior**: YES - Safety mechanism working correctly
- **Can Be Fixed**: YES - After successful training with data
- **Should Be Suppressed**: NO - Critical safety check
- **Recommendation**: None - correct behavior

## Summary

### Total Warnings: ~75-80 per training session

### Breakdown by Category
1. **TopstepX Adapter Warnings**: 70 occurrences (93% of warnings)
   - **Expected**: YES
   - **Can Fix**: NO
   - **Should Suppress**: NO (reduce log level instead)

2. **First Run Warnings**: 5-7 occurrences (7% of warnings)
   - **Expected**: YES (for first run)
   - **Can Fix**: YES (after data accumulation)
   - **Should Suppress**: NO

3. **Validation Warnings**: 2 occurrences
   - **Expected**: YES (for first run)
   - **Can Fix**: YES (after successful training)
   - **Should Suppress**: NO

### Recommendations

#### Immediate Actions
1. **Add LAB_MODE-aware log filtering** for TopstepX adapter warnings
   - Reduce log level from WARNING to DEBUG when LAB_MODE=1
   - Keep warnings visible in terminal mode

2. **Update health check thresholds** for LAB_MODE
   - Accept 60% health as "healthy" when LAB_MODE=1
   - Document expected disconnected services

3. **Add context to first-run warnings**
   - Append "(expected for first training session)" to relevant warnings
   - Update log messages to be more informative

#### Long-term Improvements
1. **Create LAB_MODE-specific health profile**
   - Define expected service states for lab mode
   - Separate health checks for Terminal vs Lab modes

2. **Add training data accumulation tracking**
   - Show progress toward minimum data requirements
   - Estimate when components will have sufficient data

3. **Enhance validation phase messaging**
   - Distinguish between "no models to test" vs "models failed tests"
   - Provide clearer next steps

### Conclusion
**Zero warnings need to be suppressed**. All warnings represent:
1. Expected behavior in Lab Mode (offline training)
2. First-run conditions (no prior data)
3. Safety mechanisms working correctly (validation gates)

The warnings are **informative and useful** for understanding:
- Training mode status
- Data availability
- Component readiness
- System health

**Action**: Improve log messaging and filtering, but do NOT suppress warnings.
