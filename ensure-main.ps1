# Normalize repository to use the 'main' branch for all work/pushes
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\ensure-main.ps1
#   powershell -ExecutionPolicy Bypass -File .\ensure-main.ps1 -PullRebase  # pulls origin/main (if exists) with --rebase before pushing
# Behavior:
#   - If HEAD is detached: creates/switches to 'main' at current commit.
#   - If on a different branch: moves/switches to 'main' at current commit.
#   - Ensures upstream to origin/main; pulls --rebase if asked and remote exists; pushes main.

param(
    [switch]$PullRebase
)

$ErrorActionPreference = 'Stop'

# Go to repo root
Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Definition)

function Is-DetachedHead {
    $ref = git symbolic-ref -q --short HEAD 2>$null
    return [string]::IsNullOrWhiteSpace($ref)
}

# Ensure hooks path to avoid pre-commit missing errors
try { git config --local core.hooksPath "scripts/hooks" | Out-Null } catch { }

# Stage and commit any outstanding changes with a generic message (no-verify)
try {
    git add -A
    $status = git status --porcelain
    if ($status) {
        $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        git commit --no-verify -m "ensure-main: $ts"
    }
} catch { }

# Create/switch 'main' at current commit
if (Is-DetachedHead) {
    Write-Host "[ensure-main] Detached HEAD detected. Creating/switching to 'main' at current commit..."
    git checkout -B main
} else {
    $cur = (git rev-parse --abbrev-ref HEAD).Trim()
    if ($cur -ne 'main') {
        Write-Host "[ensure-main] On branch '$cur'. Moving/switching to 'main' at current commit..."
        git checkout -B main
    } else {
        Write-Host "[ensure-main] Already on 'main'."
    }
}

# If origin exists, optionally pull --rebase and then push, setting upstream if needed
$originUrl = git remote get-url origin 2>$null
if ($originUrl) {
    try {
        if ($PullRebase) {
            Write-Host "[ensure-main] Pulling origin/main with --rebase (if exists)..."
            git pull --rebase origin main 2>$null
        }
    } catch { Write-Warning "[ensure-main] Pull --rebase failed: $($_.Exception.Message)" }

    try {
        # If no upstream, push with -u first
        $hasUpstream = $true
        try { $null = git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>$null } catch { $hasUpstream = $false }
        if (-not $hasUpstream) {
            Write-Host "[ensure-main] Setting upstream to origin/main..."
            git push -u origin main
        } else {
            git push origin main
        }
        Write-Host "[ensure-main] Pushed 'main' to origin."
    } catch {
        Write-Warning "[ensure-main] Push to origin failed: $($_.Exception.Message)"
    }
} else {
    Write-Host "[ensure-main] No 'origin' remote configured. Skipping pull/push."
}