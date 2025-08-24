# Auto-push script for automatically committing and pushing all changes
# Usage: powershell -ExecutionPolicy Bypass -File .\auto-push.ps1

$repoPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
$intervalSeconds = 60  # Check every 60 seconds

Write-Host "[auto-push] Monitoring for file changes... Press Ctrl+C to stop."

while ($true) {
    Set-Location $repoPath

    # Ensure hooks path points to repo scripts/hooks to avoid missing pre-commit errors
    git config --local core.hooksPath "scripts/hooks" | Out-Null

    $status = git status --porcelain
    if ($status) {
        git add -A
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        git commit --no-verify -m "Auto-push: $timestamp"
        # Push to current branch instead of hardcoded 'main'
        $branch = (git rev-parse --abbrev-ref HEAD).Trim()
        if (-not $branch) { $branch = "main" }
        git push origin $branch
        Write-Host "[auto-push] Changes pushed to '$branch' at $timestamp."
    } else {
        Write-Host "[auto-push] No changes detected."
    }
    Start-Sleep -Seconds $intervalSeconds
}
