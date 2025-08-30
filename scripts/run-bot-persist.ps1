param(
	[switch]$Prompt # Force interactive mode selection at launch
)

$ErrorActionPreference = 'Stop'

# Move to repo root (this script lives in scripts/)
$root = Split-Path -Parent $PSScriptRoot
Set-Location -Path $root

# Set env for a persistent, single-port launch
# Respect existing values; allow opt-in interactive prompts and verbose startup.

# Opt-in interactive mode selection (PROMPTS=1|true|yes) or -Prompt switch
if ($Prompt) { Set-Item Env:PROMPTS '1' }
if ($env:PROMPTS -and ($env:PROMPTS.Trim().ToLowerInvariant() -in @('1','true','yes'))) {
	Set-Item Env:SKIP_MODE_PROMPT '0'
}
elseif (-not $env:SKIP_MODE_PROMPT) {
	Set-Item Env:SKIP_MODE_PROMPT '1'
}

# Default to PAPER unless explicitly set; coerce away from stale SHADOW when not prompting
$__truthy = @('1','true','yes')
$__prompting = ($env:PROMPTS -and ($env:PROMPTS.Trim().ToLowerInvariant() -in $__truthy))
if (-not $__prompting) {
	if (-not $env:BOT_MODE) { Set-Item Env:BOT_MODE 'paper' }
	elseif ($env:BOT_MODE.Trim().ToLowerInvariant() -eq 'shadow') { Set-Item Env:BOT_MODE 'paper' }
}

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

# Live/persistent defaults: keep guards ON (historical scripts enable bypasses explicitly)
if (-not $env:BT_IGNORE_NEWS) { Set-Item Env:BT_IGNORE_NEWS '0' }
if (-not $env:BT_IGNORE_SPREAD) { Set-Item Env:BT_IGNORE_SPREAD '0' }
if (-not $env:BT_IGNORE_QUOTE_FREEZE) { Set-Item Env:BT_IGNORE_QUOTE_FREEZE '0' }

# Optional session-aware spread allowances (only used if set; safe defaults shown)
if (-not $env:SPREAD_ALLOW_ES_RTH) { Set-Item Env:SPREAD_ALLOW_ES_RTH '2' }
if (-not $env:SPREAD_ALLOW_ES_ETH) { Set-Item Env:SPREAD_ALLOW_ES_ETH '3' }
if (-not $env:SPREAD_ALLOW_NQ_RTH) { Set-Item Env:SPREAD_ALLOW_NQ_RTH '3' }
if (-not $env:SPREAD_ALLOW_NQ_ETH) { Set-Item Env:SPREAD_ALLOW_NQ_ETH '4' }

# Enable adaptive learning and safe bandit exploration in PAPER/SHADOW only
if (-not $env:RUN_LEARNING)       { Set-Item Env:RUN_LEARNING '1' }
if (-not $env:INSTANT_ALLOW_LIVE) { Set-Item Env:INSTANT_ALLOW_LIVE '0' } # prevent instant apply in LIVE

# Bandit/canary routing defaults (active in PAPER; blocked from affecting LIVE unless explicitly allowed)
if (-not $env:BANDIT_ENABLE)      { Set-Item Env:BANDIT_ENABLE '1' }
if (-not $env:CANARY_ENABLE)      { Set-Item Env:CANARY_ENABLE '1' }
if (-not $env:BANDIT_ALLOW_LIVE)  { Set-Item Env:BANDIT_ALLOW_LIVE '0' }
# Optional cadence/TTL (safe defaults)
if (-not $env:BANDIT_LOOP_MIN)    { Set-Item Env:BANDIT_LOOP_MIN '15' }
if (-not $env:BANDIT_TTL_HOURS)   { Set-Item Env:BANDIT_TTL_HOURS '4' }

# Quality-first toggles (live defaults keep scheduling/attempt caps ON)
if (-not $env:SKIP_TIME_WINDOWS) { Set-Item Env:SKIP_TIME_WINDOWS '0' }
if (-not $env:SKIP_ATTEMPT_CAPS) { Set-Item Env:SKIP_ATTEMPT_CAPS '0' }
# Backward-compat alias used by the app: treat ALL_HOURS_QUALITY like SKIP_TIME_WINDOWS (default OFF)
if (-not $env:ALL_HOURS_QUALITY) { Set-Item Env:ALL_HOURS_QUALITY '0' }

# Master bypass ET_NO_GUARD is reserved for historical/backtest only — force OFF in persistent run
Set-Item Env:ET_NO_GUARD '0'

# Optional one-switch restore: PERSIST_NO_GUARD=1|true|yes re-enables previous "no guard" defaults for persistent runs
if ($env:PERSIST_NO_GUARD -and ($env:PERSIST_NO_GUARD.Trim().ToLowerInvariant() -in @('1','true','yes'))) {
	Set-Item Env:ET_NO_GUARD '1'
	Set-Item Env:SKIP_TIME_WINDOWS '1'
	Set-Item Env:ALL_HOURS_QUALITY '1'
	Set-Item Env:SKIP_ATTEMPT_CAPS '1'
	Set-Item Env:BT_IGNORE_NEWS '1'
	Set-Item Env:BT_IGNORE_SPREAD '1'
	Set-Item Env:BT_IGNORE_QUOTE_FREEZE '1'
}

# Opt-in verbose console (VERBOSE_STARTUP=1|true|yes) — overrides default concise logging in launcher
if ($env:VERBOSE_STARTUP -and ($env:VERBOSE_STARTUP.Trim().ToLowerInvariant() -in @('1','true','yes'))) {
	Set-Item Env:APP_CONCISE_CONSOLE 'false'
}

# Invoke main launcher
& .\launch-bot.ps1