# Git Helper Scripts

This repository includes PowerShell scripts to help manage git operations and ensure consistency with the main branch workflow.

## Scripts

### push-now.ps1
One-shot git push with a custom message. Routes to main by default.

```powershell
# Basic usage
powershell -ExecutionPolicy Bypass -File .\push-now.ps1 -Message "Your commit message"

# Keep current branch (set environment variable)
$env:PREFER_MAIN="false"
powershell -ExecutionPolicy Bypass -File .\push-now.ps1 -Message "Your commit message"

# Force push with lease
powershell -ExecutionPolicy Bypass -File .\push-now.ps1 -Message "Your commit message" -Force
```

### auto-push.ps1
Automatic git push with timestamp. Always switches to main before pushing.

```powershell
powershell -ExecutionPolicy Bypass -File .\auto-push.ps1
```

### ensure-main.ps1
Normalize repository to use main branch for all work and pushes.

```powershell
# Basic setup
powershell -ExecutionPolicy Bypass -File .\ensure-main.ps1

# Setup with pull/rebase
powershell -ExecutionPolicy Bypass -File .\ensure-main.ps1 -PullRebase
```

### fix-detached-head.ps1
Fix detached HEAD state by creating a branch or switching to an existing one.

```powershell
# Create branch with auto-generated name
powershell -ExecutionPolicy Bypass -File .\fix-detached-head.ps1

# Create branch with custom name
powershell -ExecutionPolicy Bypass -File .\fix-detached-head.ps1 -Branch my-work

# Switch to existing branch (loses detached commits)
powershell -ExecutionPolicy Bypass -File .\fix-detached-head.ps1 -Checkout main
```

## Mirror Setup (Optional)

To mirror pushes to an additional remote, set these environment variables in `.env.local`:

```
GIT_EXTRA_REMOTE=backup
GIT_EXTRA_URL=https://github.com/you/your-mirror.git
```

The push scripts will automatically push to both origin and the extra remote.

## Usage Tips

1. **Start with ensure-main.ps1** to normalize your repository
2. **Use push-now.ps1** for regular commits with meaningful messages
3. **Use auto-push.ps1** for quick saves during development
4. **Use fix-detached-head.ps1** if you accidentally get into detached HEAD state

All scripts are designed to be safe and will warn you about potential issues before making changes.