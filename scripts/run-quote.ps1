Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Resolve repo root regardless of invocation location (prefer dir containing .env.local or solution at repo root)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$candidates = @(
  $scriptDir,
  (Resolve-Path (Join-Path $scriptDir '..')).Path,
  (Resolve-Path (Join-Path $scriptDir '..\..')).Path
)
$root = $scriptDir
foreach ($c in $candidates) {
  if (Test-Path (Join-Path $c '.env.local') -or Test-Path (Join-Path $c 'TopstepX.Bot.sln')) {
    $root = $c
    break
  }
}
Push-Location $root

# Load .env.local variables into current PowerShell process (robust)
$envFile = Join-Path $root '.env.local'
Write-Host "Using env file: $envFile"
if (Test-Path $envFile) {
  # Read entire file and extract TOPSTEPX_JWT first
  $raw = Get-Content $envFile -Raw
  $jwtMatch = [regex]::Match(
    $raw,
    '^\s*TOPSTEPX_JWT\s*=\s*(.+)$',
    [System.Text.RegularExpressions.RegexOptions]::Multiline
  )
  if ($jwtMatch.Success) {
    $jwt = $jwtMatch.Groups[1].Value.Trim()
    if ($jwt) {
      Set-Item -Path 'env:TOPSTEPX_JWT' -Value $jwt
      [System.Environment]::SetEnvironmentVariable('TOPSTEPX_JWT', $jwt)
      Write-Host "Loaded TOPSTEPX_JWT (len=$($jwt.Length))"
    }
  }

  # Load all other variables
  $lines = $raw -split "`r?`n"
  foreach ($line in $lines) {
    if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$') {
      $name  = $Matches[1]
      $value = $Matches[2].Trim()
      if ($name -and $value) {
        if ($name -ne 'TOPSTEPX_JWT') {
          Set-Item -Path "env:$name" -Value $value
          [System.Environment]::SetEnvironmentVariable($name, $value)
        }
      }
    }
  }
}

if (-not $env:TOPSTEPX_JWT) { throw "Set TOPSTEPX_JWT in .env.local" }

# Build and run QuoteRunner in Release
dotnet restore ".\TopstepX.Bot.sln"
dotnet build   ".\TopstepX.Bot.sln" -c Release
dotnet run --project ".\examples\QuoteRunner\QuoteRunner.csproj" -c Release

Pop-Location
