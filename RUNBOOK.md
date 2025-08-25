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
