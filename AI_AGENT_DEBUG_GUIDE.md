# 🤖 AI Agent Debugging Guide

## How to See Workflow Logs

### Current Setup
- **Workflow timeout**: 10 minutes (enough for bot startup + runtime)
- **Auto-triggers**: Every push to `fix/**`, `feature/**`, `bugfix/**` branches
- **Logs location**: GitHub Actions "Checks" tab after workflow completes

### Workflow Timeline
1. **0-2 min**: Build, Python adapter start, environment setup
2. **2-5 min**: Bot startup, ML model loading, strategy initialization
3. **5-10 min**: Bot running, processing market data
4. **10 min**: Workflow times out → Logs become available

### How to Access Logs

#### Via GitHub Web UI (Easiest)
1. Go to the PR or commit
2. Click "Checks" tab
3. Click "Run Trading Bot on Self-Hosted Runner"
4. Click "Run bot" job
5. Expand the "▶ Run Bot (Orchestrator)" step
6. **All logs are there** - scroll through to find errors

#### Via GitHub CLI (If Available)
```bash
# List recent runs
gh run list --workflow="selfhosted-bot-run.yml" --limit 3

# View logs from latest completed run
gh run view --log

# View specific run
gh run view 18597797518 --log

# Search for errors
gh run view --log | grep -i "error\|exception\|failed"
```

#### What You'll See in Logs
- ✅ Build output (first 2 minutes)
- ✅ Python adapter startup
- ✅ Bot initialization messages
- ✅ ML model loading (success/failure)
- ✅ RL policy initialization
- ✅ Strategy activation
- ✅ Market data connection status
- ✅ **All errors, exceptions, warnings**
- ✅ Stack traces for crashes

## Debugging Workflow

### Step 1: Wait for Logs (10 minutes)
After you push a commit, wait **10 minutes** for the workflow to timeout and logs to become available.

### Step 2: Analyze Real Errors
Don't guess! Read the actual logs and identify:
- Startup failures
- Missing dependencies
- Configuration errors
- Connection failures
- ML/RL initialization issues

### Step 3: Make Targeted Fixes
Fix **only the actual errors** you see in logs. Follow coding standards in `.github/copilot-instructions.md`.

### Step 4: Push and Repeat
```bash
git add .
git commit -m "fix: [describe what you fixed based on logs]"
git push
# Workflow auto-triggers, wait 10 minutes for new logs
```

## Quick Reference

| Action | Time |
|--------|------|
| Workflow starts after push | ~10 seconds |
| Build completes | ~2 minutes |
| Bot startup visible | ~3-5 minutes |
| Logs available for review | **10 minutes** (timeout) |

## Tips for Fast Iteration

1. **Be patient**: 10 minutes per iteration is necessary for full bot startup
2. **Read carefully**: One thorough log analysis > multiple guesses
3. **Fix precisely**: Surgical fixes based on real errors, not assumptions
4. **Batch related fixes**: If you see 3 related errors, fix all 3 in one commit
5. **Test locally first** (if possible): Faster feedback loop

## Common Errors to Look For

1. **Python SDK not installed**: `ModuleNotFoundError: No module named 'topstep_x'`
2. **Adapter connection failure**: `Failed to connect to Python adapter at localhost:8765`
3. **ML model files missing**: `FileNotFoundException: data/models/...`
4. **RL policy load error**: `Error loading RL policy`
5. **Strategy config invalid**: `Strategy configuration error`
6. **Market data timeout**: `WebSocket connection timeout`

## Success Criteria

The bot runs successfully when you see:
- ✅ `Trading suite initialized successfully`
- ✅ `Connected to TopstepX`
- ✅ `ML models loaded: X/X`
- ✅ `RL policies initialized`
- ✅ `Strategies activated: [list]`
- ✅ `Processing market data...`
- ✅ No ERROR or EXCEPTION messages

---

**Remember**: The 10-minute wait is necessary because the bot needs time to fully start up. Use that time to analyze logs thoroughly rather than making rushed fixes!
