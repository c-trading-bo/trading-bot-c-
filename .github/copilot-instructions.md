## Agent Rules & Eval Defaults

> **🤖 For Coding Agents**: This repository is optimized for agent development. See `README_AGENTS.md` and `CODING_AGENT_GUIDE.md` for quick start guides.

### Eval Defaults

- Prefer MES/MNQ on SIM.
- Use /Contract/available { live:false }; fall back to search.
- Only flip to EXECUTE after BarsSeen >= 10, hubs up, canTrade, contractId != null.

### Quick Loop

- Run (DRY_RUN) while iterating.
- Run (AUTO_EXECUTE) to verify flip under prechecks.
- Paste errors, apply patch based on agent brief, run again.

### Builder Agent Usage

- Use scripts/agents/api-client.md to update Core/Api/ApiClient.cs.
- Use scripts/agents/market-hub.md to improve UserHubClient & MarketHubClient.
- Use scripts/agents/verifier.md to add VerifyTodayAsync and print totals by status.

### Guardrails

- No LLM/agent in the order path. Trading loop stays pure C#.
- DRY_RUN precedence; AUTO_EXECUTE flips only after all prechecks.
- Order evidence: require orderId + GatewayUserTrade (or Trade/search row).
- Risk/tick rules: ES/MES round to 0.25, print two decimals, reject risk ≤ 0.
- kill.txt always forces DRY_RUN.
  Project

Goal: Turn strategy signals into safe, verifiable TopstepX orders and portfolio state.

Stack: C# / .NET 8 (async-first), SignalR client, REST via HttpClient.

Key refs: TopstepX + ProjectX APIs, TradingView/Quantower integration stubs, Pine v5 notes.

Live endpoints (TopstepX)

REST base: https://api.topstepx.com
User Hub (SignalR): https://rtc.topstepx.com/hubs/user
Market Hub (SignalR): https://rtc.topstepx.com/hubs/market

Guardrails (always follow)

No fills without proof. Before saying “filled”, require at least one of:

- orderId returned by the place-order call and
- a fill event from the User Hub (see Events) or
- Trade search shows an execution.

ES/MES tick size: Round any ES/MES price to 0.25. Print two decimals.
Risk math: Compute R multiple from tick-rounded values. If risk ≤ 0 → reject.
Execution switch: Respect DRY_RUN / EXECUTE=true flags. Default to dry-run unless explicitly enabled.
Idempotency: Use a unique customTag per order (e.g., S11L-YYYYMMDD-HHMMSS). Do not place duplicate orders with the same tag within a run.
Subscriptions: Guard against double event subscriptions.
Connectivity: Do not trade via VPN/VPS/remote desktop. Local device only.
Secrets: Never print tokens. Read auth from environment/Secret Manager.

Events to listen for (User Hub)

Order status stream: GatewayUserOrder (subscribe with SubscribeOrders(accountId)).
Trade fills stream: GatewayUserTrade (subscribe with SubscribeTrades(accountId)).
Log both as structured JSON with accountId, orderId, customTag, status, reason.

Logging format (structured)
[{sig}] side={BUY|SELL} symbol={ES} qty={n} entry={0.00} stop={0.00} t1={0.00} R~{0.00} tag={customTag} orderId={guid?}
ORDER account={id} status={New|Open|Filled|Cancelled|Rejected} orderId={id} reason={...}
TRADE account={id} orderId={id} fillPrice={0.00} qty={n} time={iso}

Code conventions

Async-first: static async Task Main, ConfigureAwait(false) in libraries.
HttpClient: single, reused instance; set BaseAddress = https://api.topstepx.com.
Serialization: System.Text.Json with snakeCase policy when required; tolerant to unknown fields.
Decimals: use decimal for prices/qty; no double for monetary values.
Cancellation: all I/O accepts CancellationToken.
Retries: exponential backoff for transient 5xx/408; never retry on 4xx without corrective action.

Helpers (price rounding + R multiple)
using System;
using System.Globalization;

public static class Px
{
public const decimal ES_TICK = 0.25m;
private static readonly CultureInfo Invariant = CultureInfo.InvariantCulture;

    public static decimal RoundToTick(decimal price, decimal tick = ES_TICK) =>
        Math.Round(price / tick, 0, MidpointRounding.AwayFromZero) * tick;

    public static string F2(decimal value) => value.ToString("0.00", Invariant);

    public static decimal RMultiple(decimal entry, decimal stop, decimal target, bool isLong)
    {
        var risk   = isLong ? entry - stop : stop - entry;     // must be > 0
        var reward = isLong ? target - entry : entry - target; // must be >= 0
        if (risk <= 0 || reward < 0) return 0m;
        return reward / risk;
    }

}

SignalR (User Hub) — canonical wiring
using Microsoft.AspNetCore.SignalR.Client;

public sealed class UserHubClient : IAsyncDisposable
{
private readonly HubConnection \_hub;
private bool \_wired;

    public UserHubClient(string jwt)
    {
        _hub = new HubConnectionBuilder()
            .WithUrl("https://rtc.topstepx.com/hubs/user", o =>
            {
                o.AccessTokenProvider = () => Task.FromResult(jwt);
            })
            .WithAutomaticReconnect()
            .Build();
    }

    public async Task StartAsync(long accountId, CancellationToken ct)
    {
        await _hub.StartAsync(ct);
        if (_wired) return; _wired = true;
        _hub.On<object>("GatewayUserOrder", data => Console.WriteLine($"ORDER => {data}"));
        _hub.On<object>("GatewayUserTrade", data => Console.WriteLine($"TRADE => {data}"));
        await _hub.InvokeAsync("SubscribeOrders", accountId, ct);
        await _hub.InvokeAsync("SubscribeTrades", accountId, ct);
    }

    public async ValueTask DisposeAsync() => await _hub.DisposeAsync();

}

REST — skeletons (fill fields per API docs)
public sealed class GatewayClient
{
private readonly HttpClient \_http;

    public GatewayClient(HttpClient http)
    {
        _http = http; // BaseAddress = https://api.topstepx.com
    }

    public async Task<string?> PlaceOrderAsync(object req, CancellationToken ct)
    {
        using var resp = await _http.PostAsJsonAsync("/api/Order/place", req, ct);
        resp.EnsureSuccessStatusCode();
        var json = await resp.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: ct);
        // Extract orderId from json (field name per docs)
        return json.TryGetProperty("orderId", out var id) ? id.GetString() : null;
    }

    public async Task<JsonElement> SearchOrdersAsync(object body, CancellationToken ct)
        => await PostJsonAsync("/api/Order/search", body, ct);

    public async Task<JsonElement> SearchTradesAsync(object body, CancellationToken ct)
        => await PostJsonAsync("/api/Trade/search", body, ct);

    private async Task<JsonElement> PostJsonAsync(string path, object body, CancellationToken ct)
    {
        using var resp = await _http.PostAsJsonAsync(path, body, ct);
        resp.EnsureSuccessStatusCode();
        return (await resp.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: ct)).GetProperty("data");
    }

}

Order flow contract (what to generate)

Signal → Intent: Capture sigId, side, symbol, qty, entry, stop, t1.
Normalize: round prices to tick; validate risk>0; bound qty.
Place: call /api/Order/place with customTag = sigId + timestamp.
Log: write signal + returned orderId.
Confirm: wait for either GatewayUserTrade event or trade search shows a fill; update position state.
Timeouts: cancel stale orders per strategy rules.

Prompt recipes (for Copilot Chat)
Wire the hubs: “Generate a UserHubClient as above, add reconnection logging, and deserialize GatewayUserTrade into a typed TradeEvent record.”
Place order with proof: “Add PlaceOrderAsync that returns orderId. Update the printout to include orderId and only say FILLED after a GatewayUserTrade arrives.”
Risk checks: “Insert tick rounding and R multiple checks before order placement; reject risk≤0 with a reason.”
Search today: “Add VerifyTodayAsync(accountId) that calls Order/Trade search for today’s UTC window and prints a summary.”

Style preferences
Formatting: dotnet format clean; var when obvious; file-scoped namespaces.
Errors: human-readable + structured JSON line.
Testing: fake the HTTP and Hub with interfaces; unit-test rounding + R-multiple + subscription guard.

Do/Don’t quick list
✅ Prefer decimal everywhere for prices/qty.
✅ One HttpClient reused; set DefaultRequestHeaders.Authorization = Bearer {JWT}.
✅ One-time event wiring (\_wired guard).
❌ Don’t print raw tokens or full exceptions with secrets.
❌ Don’t claim fills without orderId + fill evidence.
❌ Don’t send orders from VPN/VPS/remote environments.

Keep this file concise and link deeper docs. Copilot reads this on every chat. Add any new rules/API specifics here as you refine the bot.

## 🤖 Coding Agent Quick References

### Agent Development Setup
```bash
# Quick start for new agents
./dev-helper.sh setup && ./dev-helper.sh build

# Main entry point
dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj

# See also: README_AGENTS.md, CODING_AGENT_GUIDE.md
```

### Key Agent Guidelines
- **Analyzer warnings expected** - Don't fix unless specifically asked (~1500+ warnings normal)
- **Minimal changes only** - Surgical fixes, no rewrites
- **Use existing patterns** - Follow what's already there
- **Test frequently** - `./dev-helper.sh test` after changes
- **Focus on functionality** - Not code quality unless requested

### Agent Entry Points
- `src/UnifiedOrchestrator/` - Main application
- `src/BotCore/Services/` - Core services and DI
- `src/TopstepAuthAgent/` - API integration
- `.env` - Configuration (copy from `.env.example`)
- `./dev-helper.sh` - Development commands

Quickfix — Dashboard/HTTP not loading

- Ensure ASPNETCORE_URLS is a single port (prefer http://localhost:5050) and that no other process is listening.
- Kill stale listeners:
  - PowerShell: Get-NetTCPConnection -LocalPort 5050 -State Listen | Select -Expand OwningProcess -Unique | % { Stop-Process -Id $\_ -Force }
- Disable quick-exit: BOT_QUICK_EXIT=0 (otherwise the process exits and the port closes).
- Mode selection precedence: launcher sets BOT_MODE and SKIP_MODE_PROMPT=1; the app’s .env loader must NOT overwrite existing env vars. If missing, implement this guard before setting env from .env.
- Health probes: http://localhost:5050/healthz and /healthz/mode should respond once listening.

Developer guardrails (process & Git)

- Branch workflow: create a disposable branch per task and push upstream, e.g.:
  - git switch -c agent/feat-short-desc
  - git push -u origin HEAD
- Protect main: only merge via PR; ask for minimal patches, not rewrites.
- Patch mode prompt: “Make the smallest possible change. Return a unified diff only (no full files). Keep signatures/exports stable. Don’t touch .env, settings.json, keys/\*.”
- Do-not-touch: .env, secrets, keys, CI YAML, versioning, risk controls, auth/login code (TopstepAuthAgent, JwtCache, any Auth/\* folder).
- Pre-commit gate: use .pre-commit-config.yaml to run dotnet format/build/test and block protected files.
- Approvals: require confirmation for file writes and terminal commands when using local agents; propose a plan before executing.
- Review hunks: stage with git add -p and verify with git diff --staged before committing.

## Analyzer Guardrails — Permanent Agent Instructions

**Purpose:** Maintain strict static analyzer compliance. This repo has ~1500 documented warnings that are expected. The goal is to prevent any new warnings and to remove existing ones only when explicitly requested.

### Rules

#### No New Warnings
- Treat all analyzer warnings as errors (`-warnaserror`) during build validation.
- If a change introduces new warnings, fix them before committing.

#### No Silent Suppressions
- Do not add `#pragma warning disable` or `[SuppressMessage]` without a linked issue and justification in a code comment.

#### Respect the Baseline
- Do not attempt to "fix" the existing ~1500 warnings unless the PR explicitly requests analyzer cleanup.
- Never remove or alter documented baseline suppressions without approval.

#### Follow Existing Patterns
- Match the style, naming, and structure already in the repo.
- Use the same null-checking, logging, and exception-handling patterns as in `src/BotCore/Services/` and `src/UnifiedOrchestrator/`.

#### Validation Before Commit
Run:
```bash
./dev-helper.sh analyzer-check
```
This runs `dotnet build -warnaserror` and fails on any new warnings. Only commit after this passes.

### Acceptance Criteria
- Build passes with zero new warnings.
- No unauthorized suppressions.
- All changes align with repo coding standards in `CODING_AGENT_GUIDE.md`.
- If analyzer cleanup is requested, before/after warning counts are included in the PR description.

### 💡 How to use this with the agent every PR
Keep this section permanently in `.github/copilot-instructions.md` so the agent reads it automatically.

Add a short reminder in your PR description when assigning the agent, for example:
```
@copilot
Follow Analyzer Guardrails in .github/copilot-instructions.md.
No new warnings allowed. Run ./dev-helper.sh analyzer-check before committing.
```

## Copilot Coding Agent — Production‑Safe Task Instructions

**Scope:** You are working in a .NET 8.0 trading bot repo with strict analyzers (~1500 warnings documented as expected). Do not connect to any live APIs, including TopstepX. Only use local code, committed data files, and helper scripts. If Topstep data is needed, use only committed JSON/CSV snapshots in `data/topstep/`.

### Order of Operations

#### 1. Setup Environment

- Run `./dev-helper.sh setup`
- Run `./validate-agent-setup.sh`

#### 2. Baseline Build & Test

- Run `./dev-helper.sh build` and `./dev-helper.sh test`
- Ensure all tests pass before making changes

#### 3. Make Changes

- Follow the PR request (e.g., analyzer cleanup, dead‑code removal, refactor)
- Use patterns from `src/UnifiedOrchestrator/`, `src/BotCore/Services/`, `src/Strategies/`
- Never modify `src/TopstepAuthAgent/` to call live APIs

#### 4. Validate Changes

- Re‑run `./dev-helper.sh build` and `./dev-helper.sh test`
- Run `./dev-helper.sh analyzer-check` to ensure no new warnings
- Run `./dev-helper.sh backtest` if strategy logic changed (uses local sample data)
- Run `./dev-helper.sh riskcheck` to ensure constants match committed Topstep snapshot

#### 5. Prepare PR for Review

- Summarize changes in PR description
- Include before/after analyzer warning counts if relevant
- Confirm no new dead code or analyzer warnings

### Analyzer Guardrails

- **No new warnings** — treat all analyzer warnings as errors (`-warnaserror`)
- **No silent suppressions** — no `#pragma warning disable` or `[SuppressMessage]` without linked issue + justification
- **Respect the baseline** — do not "fix" the existing ~1500 warnings unless explicitly requested

### Risk & Policy Guardrails

- No external API calls from CI/cloud
- No trading constant changes without validating against committed snapshot data
- No bypassing branch protection — all merges require maintainer approval
- Keep changes minimal and targeted — no large rewrites unless explicitly requested

### Acceptance Criteria

- Build passes with zero new errors/warnings
- All tests pass
- Changes align with `CODING_AGENT_GUIDE.md`
- Risk constants remain compliant with committed Topstep snapshot

### 💡 PR Description Reminder

When assigning the agent, add this line so it knows to follow the above every time:

```
@copilot  
Follow the Production‑Safe Task Instructions in .github/copilot-instructions.md.  
No live API calls. No new analyzer warnings. Use helper scripts for all steps.
```
