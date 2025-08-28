# Normalize repo to use main for all work and pushes
# Usage: powershell -ExecutionPolicy Bypass -File .\ensure-main.ps1 -PullRebase

param(
    [switch]$PullRebase
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Ensure we are at repository root
Set-Location -Path $PSScriptRoot

try {
    Write-Host "Ensuring main branch setup..."
    
    # Get current branch
    $currentBranch = git rev-parse --abbrev-ref HEAD
    Write-Host "Current branch: $currentBranch"
    
    # Check if main branch exists locally
    $mainExists = git show-ref --verify --quiet refs/heads/main
    $mainExistsLocally = $LASTEXITCODE -eq 0
    
    # Check if main exists on remote
    $remoteMain = git ls-remote --heads origin main 2>/dev/null
    $mainExistsRemotely = ![string]::IsNullOrWhiteSpace($remoteMain)
    
    Write-Host "Main exists locally: $mainExistsLocally"
    Write-Host "Main exists remotely: $mainExistsRemotely"
    
    if (!$mainExistsLocally) {
        if ($mainExistsRemotely) {
            Write-Host "Creating local main branch from origin/main..."
            git checkout -b main origin/main
        } else {
            Write-Host "Creating new main branch..."
            git checkout -b main
            
            # If we have commits, we need to push to establish the remote branch
            $hasCommits = git rev-list --count HEAD 2>/dev/null
            if ($hasCommits -and $hasCommits -gt 0) {
                Write-Host "Pushing new main branch to origin..."
                git push -u origin main
            }
        }
    } else {
        # Switch to main if not already there
        if ($currentBranch -ne "main") {
            Write-Host "Switching to main branch..."
            git checkout main
        }
        
        # Set upstream if main exists remotely
        if ($mainExistsRemotely) {
            git branch --set-upstream-to=origin/main main 2>/dev/null
        }
    }
    
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to setup main branch"
    }

    # Pull and rebase if requested and remote exists
    if ($PullRebase -and $mainExistsRemotely) {
        Write-Host "Pulling and rebasing from origin/main..."
        git pull origin main --rebase
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Pull/rebase failed. You may need to resolve conflicts manually."
        }
    }

    # Show current status
    Write-Host ""
    Write-Host "Repository normalized to main branch:"
    Write-Host "Current branch: $(git rev-parse --abbrev-ref HEAD)"
    $upstream = git rev-parse --abbrev-ref '@{u}' 2>/dev/null
    if ($upstream) { Write-Host "Upstream: $upstream" }
    Write-Host ""
    Write-Host "You can now use push-now.ps1 or auto-push.ps1 for all pushes."
    
} catch {
    Write-Error "Ensure-main failed: $($_.Exception.Message)"
    Write-Host ""
    Write-Host "Manual recovery steps:"
    Write-Host "1. Check git status: git status"
    Write-Host "2. Resolve any conflicts if needed"
    Write-Host "3. Try switching to main manually: git checkout main"
    Write-Host "4. Set upstream: git branch --set-upstream-to=origin/main main"
    exit 1
}