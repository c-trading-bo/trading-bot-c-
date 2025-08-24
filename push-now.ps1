# One-shot push helper: stages, commits (no-verify), and pushes current branch
# Usage: PowerShell -ExecutionPolicy Bypass -File .\push-now.ps1 [-Message "Your commit message"]

param(
    [string]$Message = $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
)

$ErrorActionPreference = 'Stop'

$repoPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $repoPath

# Ensure hooks path points to repo scripts/hooks to avoid missing pre-commit errors
git config --local core.hooksPath "scripts/hooks" | Out-Null

# Stage all changes
git add -A

# Commit if there are changes
$status = git status --porcelain
if ($status) {
    git commit --no-verify -m "Manual push: $Message"
} else {
    Write-Host "[push-now] No changes to commit; will push anyway."
}

# Determine current branch
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if (-not $branch) { $branch = "main" }

# Push to origin on current branch
git push origin $branch
Write-Host "[push-now] Pushed branch '$branch'."