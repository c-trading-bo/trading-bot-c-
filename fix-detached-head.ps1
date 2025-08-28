# Fix detached HEAD state
# Usage: 
#   powershell -ExecutionPolicy Bypass -File .\fix-detached-head.ps1
#   powershell -ExecutionPolicy Bypass -File .\fix-detached-head.ps1 -Branch my-work
#   powershell -ExecutionPolicy Bypass -File .\fix-detached-head.ps1 -Checkout main

param(
    [string]$Branch,
    [string]$Checkout
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Ensure we are at repository root
Set-Location -Path $PSScriptRoot

try {
    # Check if we're in detached HEAD state
    $currentRef = git symbolic-ref -q HEAD 2>/dev/null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Not in detached HEAD state. Current branch: $(git rev-parse --abbrev-ref HEAD)"
        return
    }

    Write-Host "Detected detached HEAD state."
    
    # Show current commit
    $currentCommit = git rev-parse HEAD
    $shortCommit = git rev-parse --short HEAD
    Write-Host "Current commit: $shortCommit ($currentCommit)"
    
    if (![string]::IsNullOrWhiteSpace($Checkout)) {
        # Option 3: Go back to existing branch (lose detached commits)
        Write-Host "Checking out existing branch: $Checkout"
        git checkout $Checkout
        
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to checkout branch: $Checkout"
        }
        
        Write-Host "Successfully switched to $Checkout"
        Write-Warning "Any commits made in detached HEAD state have been abandoned."
        
    } else {
        # Option 1 & 2: Create branch at current commit to keep work
        if ([string]::IsNullOrWhiteSpace($Branch)) {
            $Branch = "detached-head-fix-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
        }
        
        Write-Host "Creating branch '$Branch' at current commit to preserve work..."
        git checkout -b $Branch
        
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create branch: $Branch"
        }
        
        # Set upstream to origin if the branch exists there, or push to create it
        $remoteBranch = git ls-remote --heads origin $Branch
        if ($remoteBranch) {
            Write-Host "Setting upstream to origin/$Branch..."
            git branch --set-upstream-to=origin/$Branch $Branch
        } else {
            Write-Host "Pushing new branch to origin and setting upstream..."
            git push -u origin $Branch
        }
        
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Failed to set upstream or push. You may need to push manually later."
        }
        
        Write-Host "Successfully created branch '$Branch' and switched to it."
        Write-Host "Your detached HEAD work has been preserved."
    }
    
    # Show final status
    Write-Host ""
    Write-Host "Current status:"
    Write-Host "Branch: $(git rev-parse --abbrev-ref HEAD)"
    Write-Host "Commit: $(git rev-parse --short HEAD)"
    $upstream = git rev-parse --abbrev-ref '@{u}' 2>/dev/null
    if ($upstream) {
        Write-Host "Upstream: $upstream"
    }
    
} catch {
    Write-Error "Fix detached HEAD failed: $($_.Exception.Message)"
    Write-Host ""
    Write-Host "Manual recovery steps:"
    Write-Host "1. Check current state: git status"
    Write-Host "2. Create branch manually: git checkout -b my-branch-name"
    Write-Host "3. Or checkout existing branch: git checkout main"
    exit 1
}