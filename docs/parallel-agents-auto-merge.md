# Parallel agents → auto-merge to main (safe setup)

Goal: Run two chats/agents in parallel, each landing to main automatically, without conflicts.

What stays as-is (guardrails):

- No direct pushes to main. Merge via PR only.
- Don’t edit CI YAML in this repo (protected). Use GitHub settings instead.

1. GitHub settings (one-time)

- Branch protection (Settings → Branches → main):

  - Require a pull request before merging: ON
  - Require status checks to pass: ON
  - Select checks you trust to be stable (recommend the simple “build” workflow job)
  - Require branches to be up to date before merging: ON (optional but helpful)
  - Allow auto-merge: ON
  - Block force pushes: ON

- Merge Queue (Settings → Branches → Merge queue):
  - Enable on main to serialize merges and avoid race conflicts between agents.

2. Choose required checks (keep it minimal)

- Present workflows in this repo:
  - “build” (dotnet.yml) — .NET 9 SDK, restore/build/test
  - “.NET Bot CI” (ci.yml) — .NET 8 SDK, restore/build/format/test + optional Sonar
  - “.NET Core Desktop” (dotnet-desktop.yml) — Windows packaging sample (manual use recommended)

Recommendations:

- Mark only one “build” job as Required (prefer the simpler one that always runs successfully in your org).
- If the desktop workflow isn’t used, disable it in Actions → Workflows (three-dot menu → Disable) to avoid accidental failures.
- If Sonar is enabled, ensure secrets.SONAR_TOKEN exists; otherwise don’t require that check.

3. Run two agents concurrently

- Start two chats; each will work in its own branch/PR.
- Keep their scopes separate (e.g., Chat A → src/RiskAgent/**, Chat B → src/StrategyAgent/**) to minimize conflicts.
- Use Draft PRs while iterating; Auto-merge will complete once checks pass.

4. Auto-merge options

- Built-in: enable “Auto-merge” on each PR (Squash recommended). With Merge Queue, PRs merge one-by-one.
- Optional automation: use scripts/enable-auto-merge.ps1 with GitHub CLI to auto-enable Auto-merge on PRs labeled “automerge”.

5. Local hygiene (optional but helpful)

- Enable hooks once so protected files aren’t edited accidentally:
  - git config core.hooksPath scripts/hooks
- Consider using the Python pre-commit framework to run the existing pre-commit config.

Troubleshooting

- A PR is stuck waiting for checks: confirm required checks correspond to the active workflow/job names and that disabled workflows aren’t required.
- Sonar step fails due to missing token: add SONAR_TOKEN secret or don’t require the Sonar check.
- Conflicts between agents: merge the smaller/older PR first, rebase the other branch on updated main, re-run checks.
