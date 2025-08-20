# BotCore

This project contains all shared strategy and core logic for the trading bot. It includes:
- S1–S14 strategies (AllStrategies.cs)
- Risk and model helpers (ModelsAndRisk.cs)
- Market data aggregation (BarAggregator.cs)
- API and hub clients (ApiClient.cs, HistoryApi.cs, UserHubClient.cs, MarketHubClient.cs, ExampleWireUp.cs)

All bot modules and agents should reference BotCore for shared logic.
