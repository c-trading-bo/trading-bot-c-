# 🤖 AI Agent Task: Analyze Bot Runtime Errors

## Immediate Instructions

1. **Get the latest workflow run:**
   ```bash
   gh run list --limit 1 --workflow "selfhosted-bot-run.yml"
   ```

2. **Wait for it to complete** (7 minutes total runtime)

3. **Read the complete logs:**
   ```bash
   gh run view <RUN_ID> --log
   ```

4. **Analyze for errors:**
   - TopstepX adapter connection status
   - Python SDK initialization
   - Market data streaming (ES/NQ)
   - WebSocket connection issues
   - Bot startup/initialization errors
   - Trading strategy execution errors

5. **Fix ONLY bot code** (NOT workflow setup):
   - ✅ C# bot code in `src/`
   - ✅ Python adapter in `src/adapters/topstep_x_adapter.py`
   - ✅ Configuration files
   - ❌ DO NOT touch `.github/workflows/selfhosted-bot-run.yml` (Python/SDK setup is LOCKED)

6. **Test fixes:**
   - Commit changes to `fix/bot-shutdown-issue` branch
   - Push to trigger new 7-minute workflow run
   - Repeat until bot runs without errors

## Critical Constraints

⚠️ **READ FIRST**: `.github/AI_AGENT_CONSTRAINTS.md`

### Never Modify
- Workflow Python/SDK setup (lines 40-67)
- Workflow triggers
- Workflow timeout settings

### Always Follow
- `.github/copilot-instructions.md` - Code quality standards
- Use `decimal` for all money values
- Full error handling with logging
- No stub code, no mocks, no TODOs

## Success Criteria
- ✅ Bot runs for full 7 minutes
- ✅ TopstepX adapter connects
- ✅ Market data flows
- ✅ No errors in logs
