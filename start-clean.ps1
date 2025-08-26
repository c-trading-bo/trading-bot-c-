param([switch]$ChecklistOnly)

# --- make console output compact & single-line ---
$env:Logging__Console__FormatterName = 'simple'
$env:Logging__Console__FormatterOptions__SingleLine = 'true'
$env:Logging__Console__FormatterOptions__TimestampFormat = 'HH:mm:ss '

# --- turn down global noise ---
$env:Logging__LogLevel__Default = 'Warning'
$env:Logging__LogLevel__Microsoft = 'Warning'
$env:Logging__LogLevel__Microsoft.AspNetCore = 'Warning'
$env:Logging__LogLevel__Microsoft.AspNetCore.SignalR = 'Warning'
$env:Logging__LogLevel__Microsoft.AspNetCore.Http.Connections.Client = 'Warning'

# --- keep the important categories visible ---
$env:Logging__LogLevel__Orchestrator = 'Information'
$env:Logging__LogLevel__SupervisorAgent.StatusService = 'Information'   # BOT STATUS => {...}
$env:Logging__LogLevel__BotCore.UserHubAgent = 'Information'            # UserHub/JWT/connect
$env:Logging__LogLevel__BotCore.ApiClient = 'Information'               # contract picks
$env:Logging__LogLevel__BotCore.MarketHubClient = 'Information'         # stream connects (we filter later)

# Optional: quick checklist run that exits
if ($ChecklistOnly) { $env:BOT_QUICK_EXIT='1' } else { Remove-Item Env:BOT_QUICK_EXIT -ErrorAction SilentlyContinue }

# --- what to show during BOOT (checklist) ---
$bootKeep = '(Env config|Using URL|UserHub connected|Available pick|Preflight|MODE =>|BOT STATUS)'

# --- when to switch to RUN view (any of these means we’re healthy) ---
$promote = '(MODE\s*=>\s*(LIVE|AUTOPILOT|TRADING)|Promoted|Quotes ready|Healthy|Preflight.*pass)'

# --- what to show during RUN (trading/strategy only) ---
$runKeep = '(order|fill|trade|position|risk|strategy|looking\s+for\s+a\s+trade|entry|exit|BOT STATUS|lastQuote|lastTrade)'

Write-Host ("-"*80)
Write-Host (Get-Date -Format "HH:mm:ss") "LAUNCH (clean view)…"
Write-Host ("-"*80)

# launch bot and live-filter its output
& PowerShell -ExecutionPolicy Bypass -File .\launch-bot.ps1 2>&1 |
  ForEach-Object -Begin { $phase='boot' } -Process {
    $line = $_.ToString()
    if ($phase -eq 'boot') {
      if ($line -match $bootKeep) { Write-Host $line }
      if ($line -match $promote) {
        Write-Host ("-"*80)
        Write-Host (Get-Date -Format "HH:mm:ss") "READY: switching to RUN view (quotes & trades only)…"
        Write-Host ("-"*80)
        $phase='run'
      }
    } else {
      if ($line -match $runKeep) { Write-Host $line }
    }
  }

# How to use:
# One-time health check (exits automatically):
#   .\start-clean.ps1 -ChecklistOnly
# Normal run (auto-switch to trading view, no manual input):
#   .\start-clean.ps1
