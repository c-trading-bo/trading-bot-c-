# 🤖 AI Agent Constraints - DO NOT MODIFY THESE FILES

## ❌ NEVER MODIFY THESE FILES/SECTIONS

The GitHub Actions workflow setup is **FINALIZED** and working correctly. The agent must **NEVER** modify:

### 1. Workflow Python/SDK Setup (LOCKED)
**File**: `.github/workflows/selfhosted-bot-run.yml`

**LOCKED SECTIONS** (Lines 35-67 approximately):
```yaml
# Skip .NET setup - self-hosted runner already has .NET 8.0 installed
# Skip Python setup - self-hosted runner already has Python 3.11 installed

- name: "📦 Install Python Dependencies"
  # ... (entire section is LOCKED)
```

**WHY LOCKED**:
- ✅ Python 3.11 is already installed on self-hosted runner
- ✅ SDK installation method is optimized for Windows (excludes uvloop)
- ✅ Using `--no-deps` strategy to avoid dependency conflicts
- ✅ Manual dependency installation tested and working

**If agent thinks Python/SDK needs changes**:
- ❌ DO NOT modify workflow
- ❌ DO NOT add `actions/setup-python`
- ❌ DO NOT change pip install commands
- ✅ ONLY modify C# bot code or Python adapter code

### 2. Workflow Trigger Configuration (LOCKED)
**File**: `.github/workflows/selfhosted-bot-run.yml`

**LOCKED SECTION** (Lines 11-15 approximately):
```yaml
on:
  push:
    branches-ignore:
      - 'main'
```

**WHY LOCKED**:
- ✅ Auto-triggers on ALL branches (including agent's copilot/* branches)
- ✅ Prevents main branch pollution
- ✅ Enables seamless agent iteration

### 3. Workflow Timeout (LOCKED)
**File**: `.github/workflows/selfhosted-bot-run.yml`

**LOCKED SECTION**:
```yaml
timeout-minutes: 5  # Timeout for debugging - agent can see logs after 5 min
```

**WHY LOCKED**:
- ✅ Optimized for fast iteration cycles
- ✅ Gives enough time to see startup + runtime issues
- ✅ Prevents workflow from hanging indefinitely

## ✅ WHAT AGENT CAN MODIFY

### Bot Code (ALLOWED)
- `src/UnifiedOrchestrator/**/*.cs` - All C# bot code
- `src/BotCore/**/*.cs` - Core services and interfaces
- `src/TopstepAuthAgent/**/*.cs` - TopstepX integration
- `src/Safety/**/*.cs` - Safety mechanisms
- `src/adapters/topstep_x_adapter.py` - Python adapter logic

### Configuration (ALLOWED)
- `appsettings.*.json` - Application settings
- `strategies-enabled.json` - Strategy configuration
- `.env.example` - Example environment variables

### Documentation (ALLOWED)
- `docs/**/*.md` - Documentation updates
- `README*.md` - Readme files
- `RUNBOOKS.md` - Runbook updates

## 🎯 Agent Workflow Instructions

When the agent sees errors in workflow logs:

1. **Analyze the ERROR** - Read the bot logs, not workflow setup logs
2. **Identify ROOT CAUSE** - Is it bot code or configuration?
3. **Fix BOT CODE** - Modify C# or Python adapter code
4. **Commit to SAME BRANCH** - Don't create new branches
5. **Let workflow auto-run** - It will trigger automatically
6. **Review new logs** - Wait 5 minutes and iterate

## 🚫 What Agent Should NOT Do

❌ **DO NOT** suggest installing different Python versions  
❌ **DO NOT** suggest adding `actions/setup-python@v5`  
❌ **DO NOT** suggest changing pip install commands  
❌ **DO NOT** suggest modifying workflow triggers  
❌ **DO NOT** suggest changing timeout values  
❌ **DO NOT** create new branches (work on existing branch)  
❌ **DO NOT** suggest "fixing" the workflow setup  

## ✅ What Agent SHOULD Do

✅ **DO** fix bot initialization errors in C# code  
✅ **DO** fix TopstepX adapter connection issues in Python  
✅ **DO** fix configuration errors in appsettings.json  
✅ **DO** commit fixes to the current branch  
✅ **DO** wait for workflow completion (5 minutes)  
✅ **DO** iterate based on new logs  

## 📋 Example Agent Response (CORRECT)

**User**: "The bot is crashing during TopstepX initialization"

**Agent Response**:
```
I see the error in the logs. The issue is in `src/adapters/topstep_x_adapter.py` 
at line 145 where we're calling `initialize()` with wrong parameters.

I'll fix the Python adapter code:
[makes code change]

Committed to fix/bot-shutdown-issue branch. The workflow will auto-run in 
~5 minutes and we can review the new logs.
```

## 📋 Example Agent Response (INCORRECT - DO NOT DO THIS)

**User**: "The bot is crashing during TopstepX initialization"

**Agent Response** ❌:
```
I see the error. Let me update the workflow to install the latest SDK version:
[tries to modify .github/workflows/selfhosted-bot-run.yml]

❌ WRONG - The workflow setup is LOCKED. Agent should fix bot code instead.
```

---

## 🔒 Summary

**LOCKED FILES**: Workflow setup (Python/SDK installation, triggers, timeout)  
**ALLOWED FILES**: Bot code (C#/Python), configuration, documentation  
**AGENT STRATEGY**: Fix bot code, commit to same branch, let workflow auto-run  
**ITERATION CYCLE**: 5 minutes per iteration  
