Param(
    [string]$Repo = "",
    [string]$Label = "automerge"
)

$ErrorActionPreference = 'Stop'

function Assert-Cli([string]$name, [string]$cmd) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        throw "$name is required but not found in PATH. Install it first."
    }
}

Assert-Cli "GitHub CLI" "gh"

if (-not $Repo -or $Repo -notmatch "/") {
    Write-Host "Repo not provided; attempting to detect from git origin..." -ForegroundColor Yellow
    $origin = git remote get-url origin 2>$null
    if (-not $origin) { throw "Cannot detect origin. Pass -Repo owner/name." }
    if ($origin -match "github.com[:/](?<owner>[^/]+)/(?<name>[^/.]+)") { $Repo = "$($Matches.owner)/$($Matches.name)" }
    else { throw "Unsupported origin URL: $origin" }
}

Write-Host "Scanning PRs in $Repo with label '$Label' to enable auto-merge..." -ForegroundColor Cyan

# Get open PRs with the label
$prsJson = gh pr list --repo $Repo --state open --label $Label --json number, title, mergeStateStatus, autoMergeRequest, labels  | ConvertFrom-Json
if (-not $prsJson) { Write-Host "No open PRs with label '$Label'."; exit 0 }

foreach ($pr in $prsJson) {
    $num = $pr.number
    $hasAuto = $null -ne $pr.autoMergeRequest
    if ($hasAuto) { Write-Host "PR #$num already has auto-merge enabled."; continue }
    Write-Host "Enabling auto-merge on PR #$num ($($pr.title))..." -ForegroundColor Green
    gh pr merge $num --repo $Repo --auto --squash | Out-Null
}

Write-Host "Done." -ForegroundColor Green
