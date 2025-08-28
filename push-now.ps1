# One-shot git push with message
# Usage: powershell -ExecutionPolicy Bypass -File .\push-now.ps1 -Message "note"
# Routes to main by default (set PREFER_MAIN=false to keep current branch)

param(
    [Parameter(Mandatory = $true)]
    [string]$Message,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Ensure we are at repository root
Set-Location -Path $PSScriptRoot

try {
    # Check if PREFER_MAIN is disabled
    $preferMain = $env:PREFER_MAIN
    if ([string]::IsNullOrWhiteSpace($preferMain) -or $preferMain -ne "false") {
        Write-Host "Switching to main branch (set PREFER_MAIN=false to keep current branch)..."
        
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

    # Commit changes
    Write-Host "Committing with message: $Message"
    git commit -m $Message
    
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to commit changes"
    }

    # Push to origin
    $currentBranch = git rev-parse --abbrev-ref HEAD
    Write-Host "Pushing $currentBranch to origin..."
    
    if ($Force) {
        git push --force-with-lease origin $currentBranch
    } else {
        git push origin $currentBranch
    }
    
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
        git push $extraRemote $currentBranch
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Successfully mirrored to $extraRemote"
        } else {
            Write-Warning "Failed to mirror to $extraRemote (continuing anyway)"
        }
    }

    Write-Host "Push completed successfully!"
    
} catch {
    Write-Error "Push failed: $($_.Exception.Message)"
    exit 1
}