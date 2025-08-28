# Auto git push - automatically switches to main and pushes
# Usage: powershell -ExecutionPolicy Bypass -File .\auto-push.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Ensure we are at repository root
Set-Location -Path $PSScriptRoot

try {
    Write-Host "Auto-push: switching to main branch..."
    
    # Check if main branch exists locally
    $mainExists = git show-ref --verify --quiet refs/heads/main
    if ($LASTEXITCODE -eq 0) {
        git checkout main
    } else {
        # Check if main exists on remote
        $remoteMain = git ls-remote --heads origin main
        if ($remoteMain) {
            Write-Host "Creating local main branch from origin/main..."
            git checkout -b main origin/main
        } else {
            Write-Host "No main branch found. Creating new main branch..."
            git checkout -b main
        }
    }
    
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to checkout/create main branch"
    }

    # Pull latest changes if main exists on remote
    $remoteMain = git ls-remote --heads origin main
    if ($remoteMain) {
        Write-Host "Pulling latest changes from origin/main..."
        git pull origin main --rebase
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Failed to pull/rebase from origin/main (continuing anyway)"
        }
    }

    # Add all changes
    Write-Host "Adding changes..."
    git add .
    
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to add changes"
    }

    # Check if there are changes to commit
    $status = git status --porcelain
    if ([string]::IsNullOrWhiteSpace($status)) {
        Write-Host "No changes to commit."
        return
    }

    # Generate auto-commit message
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $commitMessage = "Auto-commit: $timestamp"
    
    # Commit changes
    Write-Host "Committing with message: $commitMessage"
    git commit -m $commitMessage
    
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to commit changes"
    }

    # Push to origin
    Write-Host "Pushing main to origin..."
    git push origin main
    
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to push to origin"
    }

    # Mirror to extra remote if configured
    $extraRemote = $env:GIT_EXTRA_REMOTE
    $extraUrl = $env:GIT_EXTRA_URL
    
    if (![string]::IsNullOrWhiteSpace($extraRemote) -and ![string]::IsNullOrWhiteSpace($extraUrl)) {
        Write-Host "Mirroring to extra remote: $extraRemote"
        
        # Add remote if it doesn't exist
        $remotes = git remote
        if ($remotes -notcontains $extraRemote) {
            git remote add $extraRemote $extraUrl
        }
        
        # Push to extra remote
        git push $extraRemote main
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Successfully mirrored to $extraRemote"
        } else {
            Write-Warning "Failed to mirror to $extraRemote (continuing anyway)"
        }
    }

    Write-Host "Auto-push completed successfully!"
    
} catch {
    Write-Error "Auto-push failed: $($_.Exception.Message)"
    exit 1
}