# Topstep Trading Bot Workflow

## Overview

This GitHub Actions workflow (`topstep-bot.yml`) automates the execution of the Topstep trading bot on a self-hosted runner. The workflow handles Python environment setup, dependency installation, credential management, and log collection.

## Files

- **`.github/workflows/topstep-bot.yml`** - GitHub Actions workflow definition
- **`bot.py`** - Python wrapper script that launches the C# trading bot

## Features

### ✅ Self-Hosted Runner Execution
- Runs on your own infrastructure for security and control
- Avoids exposing trading credentials to GitHub-hosted runners
- Supports both Windows and Linux self-hosted runners

### ✅ Python Environment Management
- Automatically installs Python 3.11
- Upgrades pip, setuptools, and wheel
- Installs dependencies from `requirements.txt`
- Verifies Topstep SDK (project-x-py) installation

### ✅ Credential Management
- Loads API credentials from GitHub Secrets
- Validates credentials before launching the bot
- Never logs sensitive credential values

### ✅ Comprehensive Logging
- Logs to both console and file
- Timestamped log files for each run
- Uploads log files as workflow artifacts
- Retains logs for 30 days

### ✅ Safety Features
- Defaults to DRY-RUN mode (no live trading)
- Requires manual confirmation for live trading
- Validates environment before starting bot
- Graceful shutdown handling

## Setup Instructions

### 1. Configure GitHub Secrets

Add the following secrets in your GitHub repository:

**Settings → Secrets and variables → Actions → New repository secret**

| Secret Name | Description | Required |
|-------------|-------------|----------|
| `TOPSTEPX_API_KEY` | Your TopstepX API key | ✅ Yes |
| `TOPSTEPX_USERNAME` | Your TopstepX username | ✅ Yes |
| `TOPSTEPX_ACCOUNT_ID` | Your TopstepX account ID | Optional |

**Important:** Never commit these credentials to your repository!

### 2. Set Up Self-Hosted Runner

#### Windows Runner

1. Go to **Settings → Actions → Runners → New self-hosted runner**
2. Follow the instructions to download and configure the runner
3. Ensure the following are installed on the runner:
   - PowerShell 7+
   - .NET SDK 8.0+
   - Python 3.11+ (or the workflow will install it via Chocolatey)
   - Git

#### Linux Runner

1. Go to **Settings → Actions → Runners → New self-hosted runner**
2. Follow the instructions to download and configure the runner
3. Ensure the following are installed on the runner:
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3.11 python3.11-venv python3.11-dev dotnet-sdk-8.0 git
   ```

### 3. Verify Runner Connectivity

Test your self-hosted runner with the existing test workflow:

```bash
# Trigger the self-hosted test workflow
# Go to Actions → Self-Hosted Runner Test → Run workflow
```

## Usage

### Manual Execution

#### Option 1: Via GitHub UI

1. Go to **Actions** tab
2. Select **"🤖 Topstep Trading Bot"** workflow
3. Click **"Run workflow"**
4. Configure options:
   - **dry_run**: ✅ Enabled (recommended) - No live trading
   - **dry_run**: ❌ Disabled - ⚠️ **LIVE TRADING MODE**
   - **log_level**: INFO (default)
5. Click **"Run workflow"**

#### Option 2: Via GitHub CLI

```bash
# Dry-run mode (safe)
gh workflow run topstep-bot.yml -f dry_run=true

# Live trading mode (⚠️ USE WITH CAUTION)
gh workflow run topstep-bot.yml -f dry_run=false
```

### Scheduled Execution

The workflow is configured to run automatically Monday-Friday at market open:

- **Schedule**: 1:20 PM UTC (9:20 AM ET)
- **Days**: Monday through Friday
- **Mode**: DRY-RUN by default (for safety)

To enable live trading on schedule, modify the workflow:

```yaml
# IMPORTANT: Only do this if you understand the risks!
- name: "🤖 Launch Trading Bot"
  run: |
    # Remove the dry-run default for scheduled runs
    if ($dryRun -eq "true" -or $dryRun -eq "") {
      # Change this condition to only add --dry-run when explicitly requested
      $botArgs += "--dry-run"
    }
```

## Workflow Steps

The workflow executes the following steps in order:

1. **📥 Checkout Repository** - Clones the code to the runner
2. **🐍 Setup Python 3.11** - Installs/verifies Python 3.11
3. **📦 Install pip and upgrade** - Updates pip to latest version
4. **📦 Install Python dependencies** - Installs from requirements.txt
5. **🔍 Verify Topstep SDK** - Confirms project-x-py is available
6. **🔧 Setup Environment Variables** - Configures credentials from secrets
7. **📁 Create Logs Directory** - Prepares log storage
8. **🤖 Launch Trading Bot** - Runs bot.py which launches the C# bot
9. **📤 Upload Bot Logs** - Saves logs as workflow artifacts
10. **📊 Execution Summary** - Displays final status and log summary

## Log Files

### Accessing Logs

1. **During Execution**: View real-time logs in the workflow run
2. **After Completion**: Download artifacts from the workflow run:
   - Go to workflow run page
   - Scroll to **Artifacts** section
   - Download `bot-logs-{run_id}.zip`

### Log File Format

Logs are timestamped and include:
- Environment validation results
- SDK initialization status
- Bot startup messages
- Trading activity (in DRY-RUN mode, simulated)
- Error messages and stack traces
- Shutdown status

Example log file name: `logs/bot-20241014-132045.log`

### Log Retention

- **Artifact Retention**: 30 days (configurable in workflow)
- **Local Retention**: Managed by runner (cleanup recommended)

## Troubleshooting

### Bot Fails to Start

**Problem**: Workflow fails at "Launch Trading Bot" step

**Solutions**:
1. Check that all required secrets are set correctly
2. Verify self-hosted runner has .NET SDK installed:
   ```powershell
   dotnet --version
   ```
3. Check runner logs for detailed error messages

### Python SDK Not Found

**Problem**: Workflow fails at "Verify Topstep SDK" step

**Solutions**:
1. Check that requirements.txt includes `project-x-py[all]>=3.5.0`
2. Verify runner has internet access to install packages
3. Try installing SDK manually on runner:
   ```bash
   python3.11 -m pip install 'project-x-py[all]>=3.5.0'
   ```

### Missing Credentials

**Problem**: Workflow fails with "Missing required environment variables"

**Solutions**:
1. Verify secrets are created in GitHub repository settings
2. Check secret names match exactly (case-sensitive):
   - `TOPSTEPX_API_KEY`
   - `TOPSTEPX_USERNAME`
3. Re-add secrets if they were recently updated

### Self-Hosted Runner Offline

**Problem**: Workflow queued but never starts

**Solutions**:
1. Check runner service is running:
   ```powershell
   # Windows
   Get-Service actions.runner.*
   
   # Linux
   systemctl status actions.runner.*
   ```
2. Restart runner service if needed
3. Check runner logs for connectivity issues

### Bot Runs Too Long

**Problem**: Workflow times out after 8 hours

**Solutions**:
1. Increase `timeout-minutes` in workflow:
   ```yaml
   timeout-minutes: 960  # 16 hours
   ```
2. Check if bot is stuck in an infinite loop
3. Review bot logs for performance issues

## Security Considerations

### ✅ Best Practices

- ✅ **Never commit credentials** to repository
- ✅ **Use GitHub Secrets** for all sensitive data
- ✅ **Run on self-hosted runner** for security control
- ✅ **Default to DRY-RUN mode** for safety
- ✅ **Review logs** before enabling live trading
- ✅ **Monitor workflow runs** regularly

### ⚠️ Important Warnings

- ⚠️ **Self-hosted runners** should be on trusted, secure networks
- ⚠️ **Don't use public runners** for live trading (credentials exposure risk)
- ⚠️ **Review all code changes** before running in live mode
- ⚠️ **Test thoroughly** in DRY-RUN mode before going live
- ⚠️ **Have kill switch ready** (create `kill.txt` to force DRY-RUN)

### 🔒 Kill Switch

To immediately force the bot into DRY-RUN mode:

```bash
# Create kill.txt in repository root
touch kill.txt
git add kill.txt
git commit -m "Emergency: Force DRY-RUN mode"
git push
```

The bot will detect this file and automatically enter safe mode.

## Customization

### Modify Schedule

Edit the cron expression in the workflow:

```yaml
schedule:
  # Current: Mon-Fri at 1:20 PM UTC (9:20 AM ET)
  - cron: '20 13 * * 1-5'
  
  # Example: Mon-Fri at 2:00 PM UTC (10:00 AM ET)
  - cron: '0 14 * * 1-5'
```

Use [crontab.guru](https://crontab.guru/) to validate cron expressions.

### Change Python Version

Update the version in the workflow:

```yaml
env:
  PYTHON_VERSION: "3.12"  # Change to desired version
```

And update installation commands to use `python3.12`.

### Adjust Log Retention

Change artifact retention period:

```yaml
- name: "📤 Upload Bot Logs"
  uses: actions/upload-artifact@v4
  with:
    retention-days: 60  # Change from 30 to 60 days
```

### Add Custom Environment Variables

Add to the "Setup Environment Variables" step:

```yaml
- name: "🔧 Setup Environment Variables"
  run: |
    "CUSTOM_VAR=value" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
```

## Command Line Usage

You can also run `bot.py` directly on the runner:

```bash
# Show help
python3 bot.py --help

# Run in dry-run mode
python3 bot.py --dry-run

# Run with custom log file
python3 bot.py --log-file /path/to/custom.log

# Run in live mode (⚠️ CAUTION)
python3 bot.py
```

## Related Documentation

- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Repository structure and components
- [CODING_AGENT_GUIDE.md](CODING_AGENT_GUIDE.md) - Development guide
- [.github/copilot-instructions.md](.github/copilot-instructions.md) - Production guardrails
- [.github/workflows/README.md](.github/workflows/README.md) - Other workflows

## Support

For issues or questions:

1. Check workflow run logs for detailed error messages
2. Review this documentation for common solutions
3. Check [GitHub Issues](https://github.com/Quotraders/QBot/issues) for similar problems
4. Create a new issue with:
   - Workflow run URL
   - Error messages
   - Steps to reproduce
   - Log file (if available)

## Changelog

### 2024-10-14 - Initial Release
- Created `topstep-bot.yml` workflow
- Created `bot.py` launcher script
- Added Python 3.11 support
- Configured self-hosted runner execution
- Implemented comprehensive logging
- Added DRY-RUN mode by default
- Configured artifact upload for logs
