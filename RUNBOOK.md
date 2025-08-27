# Junie Bot Runbook

## Env

TOPSTEPX_JWT, TOPSTEPX_ACCOUNT_ID, TOPSTEPX_SYMBOLS, LIVE_ORDERS, KILL_SWITCH, PANIC_FLATTEN, BOT_ALERT_WEBHOOK

## Start/Stop

- Start: service starts on boot (systemd/NSSM). Logs → ./logs, state → ./state
- Stop: set `ROUTE_PAUSE=1`, then `KILL_SWITCH=1`, then stop service.

## Flags

- ROUTE_PAUSE=1/0 — pause/resume routing (strategies still run)
- KILL_SWITCH=1 — cancel all & stop emits
- PANIC_FLATTEN=1 — market out everything
- ALWAYS_ON=1 — gates become scoring/size-only

## Rollover

- Automatic; if needed: call `/api/rollover?root=ES` or set env `FORCE_ROLLOVER=ES`.

## Logs/State

- Logs: `./logs/bot-YYYYMMDD.log` (30d keep; >7d gz)
- State: `./state/` recent CIDs, last signals, EOD journal.

## Health

- `/healthz` — JSON summary; warns on DST boundary window.
- `/metrics` — Prometheus (quotes age, hub state, p50/p95 order latency). Note: metrics endpoint may be disabled if Prometheus wiring is not enabled in the current build.

## Alerts

- Discord/Slack webhook via `BOT_ALERT_WEBHOOK`. Alerts on breakers, rejects, reconnect loops, staleness.

## Who to page

- On breakers/reject storms: @you
- On reconnect loops > 5 in 5m: @netops

## Operational Notes

- Daily breaker: emits status `gate.daily` and pauses routing when net PnL (fees included) exceeds max daily loss.
- Bar gaps: Supervisor backfills missing minute bars and suppresses the first bar after a gap to avoid flat indicators.
- Router safety: rate-limited, idempotent CIDs, and auto-pause after reject bursts.
- EOD: Writes `state/eod_journal.jsonl` and triggers a reset hook.
- Watchdog: Exits if RSS/thread thresholds are exceeded (service manager should restart).


## Launch and Zero-Downtime Upgrades

Launch (once)
- Ensure .env.local contains real creds (examples):
  - TOPSTEPX_JWT=... (or TOPSTEPX_USERNAME + TOPSTEPX_API_KEY)
  - TOPSTEPX_ACCOUNT_ID=123456
  - TOPSTEPX_SYMBOLS=ES,NQ
  - BOT_ALERT_WEBHOOK=https://...
- Launch in terminal (clean view recommended; bot on port 5000, starts in SHADOW and auto-promotes to LIVE when healthy + lease):
  - PowerShell (compact checklist then quotes/trades only):
    - .\start-clean.ps1
    - .\start-clean.ps1 -ChecklistOnly   # quick health check that exits
  - CMD / double-click:
    - start-clean.cmd
- Verbose launch (original scripts, if you prefer full logs):
  - powershell -ExecutionPolicy Bypass -File .\launch-bot.ps1
  - launch-bot.cmd
- Start the auto-upgrader sidecar (builds/tests, runs vNext on 5001):
  - powershell -ExecutionPolicy Bypass -File .\launch-updater.ps1

Quick smoke check
- Invoke-RestMethod http://localhost:5000/healthz        | ConvertTo-Json -Depth 5  # ok:true
- Invoke-RestMethod http://localhost:5000/healthz/mode   # "SHADOW" → flips to "LIVE"

How upgrades happen (no clicks, no downtime)
- Commit/paste new code into the repo.
- UpdaterAgent automatically:
  - builds (and tests/replays if enabled),
  - launches the new build (vNext) on port 5001 in SHADOW,
  - waits for /healthz to pass (dry-run window + healthy streak),
  - calls /demote on the current bot (5000) → old enters DRAIN and releases state/live.lock,
  - vNext acquires the lease, auto-promotes to LIVE, and starts routing,
  - old keeps managing any open positions until flat, then exits.
- If anything fails, Updater does not switch (you stay on the current live build).

What guarantees no double orders
- Router hard-gate: places only when Mode == LIVE and the process holds the lease (state/live.lock).
- DRAIN mode: old build stops opening new parents during handoff but still manages exits.

Handy checks & controls
- Status:
  - Invoke-RestMethod http://localhost:5000/healthz/mode  # current live
  - Invoke-RestMethod http://localhost:5001/healthz/mode  # new build in shadow/live
- Manual override (rarely needed):
  - Invoke-RestMethod -Method POST http://localhost:5000/demote   # put current into DRAIN/SHADOW
  - Invoke-RestMethod -Method POST http://localhost:5001/promote  # force new build LIVE (only if it holds lease)

Reminders
- Run on your local device (Topstep rule). No VPN/VPS/remote desktop for live routing.
- Keep BOT_QUICK_EXIT=0.
- Ports: live bot on 5000, vNext on 5001 (configurable via ASPNETCORE_URLS or appsettings Updater:ShadowPort/LivePort).
- /metrics exposes quotes age, hubs, order latency when enabled.

Git push helpers
- One-shot: powershell -ExecutionPolicy Bypass -File .\push-now.ps1 -Message "note"
- Auto: powershell -ExecutionPolicy Bypass -File .\auto-push.ps1
- Mirror to another remote (optional): set in .env.local
  - GIT_EXTRA_REMOTE=backup
  - GIT_EXTRA_URL=https://github.com/you/your-mirror.git
  The scripts will push to origin and also mirror to the configured extra remote.

Everything on main (recommended)
- Normalize your repo to use main for all work and pushes:
  - powershell -ExecutionPolicy Bypass -File .\ensure-main.ps1 -PullRebase
  - Then push normally: powershell -ExecutionPolicy Bypass -File .\push-now.ps1 -Message "note"
- push-now.ps1 routes to main by default (set PREFER_MAIN=false to keep current branch).
- auto-push.ps1 also auto-switches to main before pushing.

Fix detached HEAD
- Symptom: git status shows `HEAD (no branch)` or `HEAD detached at <commit>`.
- Safe fix (keeps current work): create a branch at the current commit and set upstream.
  - powershell -ExecutionPolicy Bypass -File .\fix-detached-head.ps1
  - or choose a name: powershell -ExecutionPolicy Bypass -File .\fix-detached-head.ps1 -Branch my-work
- If you just want to go back to an existing branch (and you don’t need to keep detached commits):
  - powershell -ExecutionPolicy Bypass -File .\fix-detached-head.ps1 -Checkout main
- After fixing, you can push as usual with push-now.ps1 or auto-push.ps1.
