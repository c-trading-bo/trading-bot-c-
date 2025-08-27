# One-shot push helper: stages, commits (no-verify), and pushes to main by default
# Usage: PowerShell -ExecutionPolicy Bypass -File .\push-now.ps1 [-Message "Your commit message"]
# Behavior:
#  - If HEAD is detached or on another branch and PREFER_MAIN is enabled (default), it switches/creates 'main' at current commit.
#  - Pulls --rebase origin/main if upstream exists (best-effort), then pushes to origin main and optional mirror.

param(
    [string]$Message = $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
)

$ErrorActionPreference = 'Stop'

$repoPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $repoPath

# Ensure hooks path points to repo scripts/hooks to avoid missing pre-commit errors
try { git config --local core.hooksPath "scripts/hooks" | Out-Null } catch { }

# Stage all changes
git add -A

# Commit if there are changes
$status = git status --porcelain
if ($status) {
    git commit --no-verify -m "Manual push: $Message"
} else {
    Write-Host "[push-now] No changes to commit; will push anyway."
}

# Helper: check if detached
function Is-DetachedHead {
    $ref = git symbolic-ref -q --short HEAD 2>$null
    return [string]::IsNullOrWhiteSpace($ref)
}

# Prefer routing to main unless explicitly disabled
$prefer = $env:PREFER_MAIN
$preferMain = $true
if ($prefer) { $preferMain = ($prefer.Trim().ToLowerInvariant() -in @('1','true','yes')) }

if (Is-DetachedHead) {
    if ($preferMain) {
        Write-Host "[push-now] Detached HEAD. Creating/switching to 'main' at current commit..."
        git checkout -B main
    } else {
        Write-Host "[push-now] Detached HEAD; PREFER_MAIN disabled. Proceeding without switching."
    }
} else {
    $current = (git rev-parse --abbrev-ref HEAD).Trim()
    if ($preferMain -and $current -ne 'main') {
        Write-Host "[push-now] On '$current'. Switching to 'main' at current commit..."
        git checkout -B main
    }
}

# Final branch target
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if (-not $branch) { $branch = 'main' }

# If pushing main to origin, try to pull --rebase first and ensure upstream
$originUrl = git remote get-url origin 2>$null
if ($originUrl -and $branch -eq 'main') {
    try { git pull --rebase origin main 2>$null } catch { Write-Host "[push-now] pull --rebase skipped/failed." }
    # Ensure upstream exists, set if missing
    $hasUpstream = $true
    try { $null = git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>$null } catch { $hasUpstream = $false }
    if (-not $hasUpstream) {
        git push -u origin main
        Write-Host "[push-now] Set upstream to origin/main."
    } else {
        git push origin main
    }
    Write-Host "[push-now] Pushed 'main' to origin."
} else {
    git push origin $branch
    Write-Host "[push-now] Pushed branch '$branch' to origin."
}

# Optionally push to an extra remote if configured via env vars
$extraRemote = $env:GIT_EXTRA_REMOTE
$extraUrl = $env:GIT_EXTRA_URL
if ($extraRemote -and $extraUrl) {
    try {
        # Ensure remote exists and points to the URL
        $existing = git remote get-url $extraRemote 2>$null
        if (-not $existing) {
            git remote add $extraRemote $extraUrl
        } else {
            if ($existing.Trim() -ne $extraUrl.Trim()) {
                git remote set-url $extraRemote $extraUrl
            }
        }
        git push $extraRemote $branch
        Write-Host "[push-now] Mirrored branch '$branch' to $extraRemote."
    } catch {
        Write-Warning "[push-now] Extra remote push failed: $($_.Exception.Message)"
    }
}