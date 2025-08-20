# PlaceOrderTest

## Safety Guardrails & Execution Rules

- **DRY_RUN is default:** Bot will not place live orders unless AUTO_EXECUTE is explicitly enabled and all prechecks pass.
- **Risk checks:** All orders require tick rounding, R multiple > 0, and position/risk limits enforced before placement.
- **Order evidence:** No fills are confirmed without orderId and either a fill event or trade search result.
- **Kill switch:** Presence of `kill.txt` always forces DRY_RUN mode.
- **No secrets in logs:** API keys/tokens are never printed or logged.
- **Local only:** Bot must run on local device, not VPN/VPS/remote desktop.
- **Idempotency:** Unique customTag per order; no duplicate orders in a run.
- **Diagnostics:** All order/trade events are logged for audit and debugging.

Refer to `copilot-instructions.md` for full agent and safety rules.
