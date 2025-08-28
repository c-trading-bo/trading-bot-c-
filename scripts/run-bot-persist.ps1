$ErrorActionPreference = 'Stop'

# Move to repo root (this script lives in scripts/)
$root = Split-Path -Parent $PSScriptRoot
Set-Location -Path $root

# Set env for a persistent, single-port launch
Set-Item Env:SKIP_MODE_PROMPT '1'
Set-Item Env:BOT_MODE 'paper'
Set-Item Env:BOT_QUICK_EXIT '0'
Set-Item Env:SKIP_CONNECTIVITY_PROBE '1'
if (-not $env:ASPNETCORE_URLS) { Set-Item Env:ASPNETCORE_URLS 'http://localhost:5050' }
if (-not $env:TOPSTEPX_API_BASE)   { Set-Item Env:TOPSTEPX_API_BASE   'https://api.topstepx.com' }
if (-not $env:TOPSTEPX_RTC_BASE)   { Set-Item Env:TOPSTEPX_RTC_BASE   'https://rtc.topstepx.com' }
if (-not $env:RTC_USER_HUB)        { Set-Item Env:RTC_USER_HUB        'https://rtc.topstepx.com/hubs/user' }
if (-not $env:RTC_MARKET_HUB)      { Set-Item Env:RTC_MARKET_HUB      'https://rtc.topstepx.com/hubs/market' }
if (-not $env:AUTO_GO_LIVE)        { Set-Item Env:AUTO_GO_LIVE        'false' }
if (-not $env:AUTO_STICKY_LIVE)    { Set-Item Env:AUTO_STICKY_LIVE    'false' }
if (-not $env:AUTO_MIN_HEALTHY_PASSES) { Set-Item Env:AUTO_MIN_HEALTHY_PASSES '3' }

# Invoke main launcher
& .\launch-bot.ps1