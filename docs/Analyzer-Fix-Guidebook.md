# Analyzer Fix Guidebook (v1.0)

## How to use
Before fixing any compiler/analyzer error:
- Fix **code** — never weaken analyzers or add suppressions.
- Prefer **immutable-by-default** domain models (collections exposed read-only).
- Move thresholds/fees/timeouts/windows to **strongly-typed config** with validation (no magic numbers).

---

## Priority order
1) Correctness & invariants: S109, CA1062, CA1031/S2139, CS0165, CS0200, CS0201, CS0818, CS0103/CS0117/CS1501/CS1929  
2) API & encapsulation: CA2227, CA1051/S1104, CA1002, CA1034, CA2234, CA1003  
3) Logging & diagnosability: CA1848, CA2254, S1481, S1541  
4) Globalization & string ops: CA1305/1307/1304/1308/1310/1311  
5) Async/Dispose/Resource safety & perf: CA2000, CA1063, CA1854, CA1869, S2681, S3358, S1066, S2589, S1905  
6) Style/micro-perf: CA1822/S2325, CA1707, S6608/S6667/S6580, CA1860, CA1826, S4487, CA1819

---

## "Always" fixes
- **CA2227 / CS0200 (collection mutability):** Domain objects expose collections read-only. Use backing + `Replace*` and update call sites.
  ```csharp
  private readonly List<Trade> _trades = new();
  public IReadOnlyList<Trade> Trades => _trades;
  public void ReplaceTrades(IEnumerable<Trade> items)
  { _trades.Clear(); if (items != null) _trades.AddRange(items); }
  ```

- **CA1062 (public arg null-guards):**
  ```csharp
  public Result Execute(OrderSpec spec)
  { if (spec is null) throw new ArgumentNullException(nameof(spec)); /*...*/ }
  ```

- **CA1848 / CA2254 (logging):** Structured logging only (templates or LoggerMessage). No interpolation.
  ```csharp
  _log.LogInformation("Filled {Symbol} at {Price}", fill.Symbol, fill.Price);
  ```

- **CA2000 / CA1063 (dispose):** Own it → dispose it. using/await using; implement proper Dispose/DisposeAsync.

- **CS0165 / CS0818:** Initialize locals; don't use var without initializer.

- **CS0103 / CS0117 / CS1501 / CS1929:** Use real APIs (correct using/type/member/signature/receiver). No dummy members/overloads.

## "Depends" fixes (decision rules)

- **S109 magic numbers:** Usually move to config or named constants.
  Allowed inline: obvious sentinel values (-1,0,1) and explicit time helpers (TimeSpan.FromSeconds(5)).

- **Globalization:** Protocols/tickers/logs → InvariantCulture + StringComparison.Ordinal. UI text → CurrentCulture. Normalize keys with ToUpperInvariant().

- **S1541/S3358/S1066/S2589:** Refactor for clarity. Exception: proven hot-paths—document and micro-benchmark.

- **CA1822 / S2325:** Make static if no instance state. Exceptions: interface/override or imminent instance use (document).

- **CA1002:** Expose IReadOnlyList<T>/IEnumerable<T> not List<T>. Exception: internal APIs where controlled mutation is intended.

## Recipes by rule (most common)

- **S109:** Move numeric thresholds/fees/leverage/timeouts to strongly-typed options bound from config and validated on startup.

- **CA1062:** Null-guard all public/entry points and external integration boundaries.

- **CA1031 / S2139:** Catch specific exceptions; log context; rethrow or return controlled failure. No empty catch.

- **CA2227 / CS0200:** Collections stay read-only; use Replace* methods and fix call sites. DTOs can use init + map to domain.

- **CA1848 / CA2254:** Use ILogger templates or LoggerMessage source-gen; never interpolate strings in log calls.

- **CA1305/1307/1304/1308/1310/1311:** Always pass CultureInfo and StringComparison.

- **CA2000 / CA1063:** Wrap disposables; implement disposal pattern; don't create/dispose HttpClient per call (use factory/singleton).

- **CA1854:** Prefer TryGetValue to avoid double lookup.

- **CA1869:** Reuse a single JsonSerializerOptions instance.

- **S2681 / S3358 / S1066 / S2589 / S1905:** Braces always; un-nest ternaries; merge simple if's; remove constant conditions and redundant casts.

- **CA1822 / S2325:** Mark helper methods static.

- **CA1707:** Public identifiers in PascalCase (tests can use [DisplayName] for readability).

## Examples

### DTO vs domain for CS0200/CA2227

```csharp
public sealed record StrategyDto { public required string Name { get; init; }
  public required List<string> Symbols { get; init; } = new(); }

public sealed class Strategy {
  private readonly List<string> _symbols = new();
  public IReadOnlyList<string> Symbols => _symbols;
  public void ReplaceSymbols(IEnumerable<string> items)
  { _symbols.Clear(); _symbols.AddRange(items ?? Array.Empty<string>()); }
  public static Strategy FromDto(StrategyDto dto) { var s=new Strategy(); s.ReplaceSymbols(dto.Symbols); return s; }
}
```

### Globalization

```csharp
var s = amount.ToString(CultureInfo.InvariantCulture);
if (symbol.StartsWith("ES", StringComparison.Ordinal)) { /*...*/ }
var key = symbol.ToUpperInvariant();
```

### Resilience boundary (CA1031)

```csharp
while (!ct.IsCancellationRequested) {
  try { await _orchestrator.PollAsync(ct); }
  catch (ExchangeThrottleException ex) { _log.LogWarning(ex,"Throttle: backing off"); await Task.Delay(TimeSpan.FromSeconds(2), ct); }
  catch (Exception ex) { _log.LogError(ex,"Fatal loop error; stopping"); await _risk.KillSwitchAsync("fatal-loop", ct); break; }
}
```

## Pre-commit self-checks (run locally)
```bash
# New public setters (bad—esp. collections)
rg -n 'public\s+[^\{]+\{\s*get;\s*set;\s*\}\s*$'
rg -n '(List|Dictionary|I(ReadOnly)?(List|Dictionary))<.+>\s*\{\s*get;\s*set;'

# Magic numbers (skip config/markdown)
rg -n --glob '!**/*.json' --glob '!**/*.md' '[^A-Za-z0-9_](\d{2,}|0?\.\d{2,})[^A-Za-z0-9_]'

# Swallowed exceptions
rg -n 'catch\s*\(\s*Exception[^\)]*\)\s*\{(\s*//.*)?\s*\}'
```

## PR author checklist (must tick all)

- dotnet build -warnaserror green; dotnet test green
- No posture edits; no suppressions
- No new public setters on collections/domain state
- Sonar Quality Gate PASS (Reliability A); duplication within policy
- Attach tools/analyzers/current.sarif + short "fixed rules → how" summary

---