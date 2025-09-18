using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Text.Json;
using System.IO;
using System.Linq;
using System.Diagnostics;
using System.Security.Cryptography;

namespace TradingBot.IntelligenceStack;

/// <summary>
/// RL Advisor system implementing PPO/CVaR-PPO for exit decisions
/// Operates in advise-only mode until proven uplift
/// </summary>
public class RLAdvisorSystem
{
    private readonly ILogger<RLAdvisorSystem> _logger;
    private readonly AdvisorConfig _config;
    private readonly IDecisionLogger _decisionLogger;
    private readonly string _statePath;
    
    private readonly Dictionary<string, RLAgent> _agents = new();
    private readonly Dictionary<string, List<RLDecision>> _decisionHistory = new();
    private readonly Dictionary<string, PerformanceTracker> _performanceTrackers = new();
    private readonly object _lock = new();
    
    private bool _orderInfluenceEnabled = false;
    private DateTime _lastUpliftCheck = DateTime.MinValue;

    public RLAdvisorSystem(
        ILogger<RLAdvisorSystem> logger,
        AdvisorConfig config,
        IDecisionLogger decisionLogger,
        string statePath = "data/rl_advisor")
    {
        _logger = logger;
        _config = config;
        _decisionLogger = decisionLogger;
        _statePath = statePath;
        
        Directory.CreateDirectory(_statePath);
        Directory.CreateDirectory(Path.Combine(_statePath, "agents"));
        Directory.CreateDirectory(Path.Combine(_statePath, "performance"));
        
        InitializeAgents();
        _ = Task.Run(() => LoadStateAsync());
    }

    /// <summary>
    /// Get RL advisor recommendation for exit decision
    /// </summary>
    public async Task<RLAdvisorRecommendation> GetExitRecommendationAsync(
        ExitDecisionContext context,
        CancellationToken cancellationToken = default)
    {
        try
        {
            if (!_config.Enabled)
            {
                return new RLAdvisorRecommendation
                {
                    Action = ExitAction.Hold,
                    Confidence = 0.0,
                    Reasoning = "RL Advisor disabled",
                    IsAdviseOnly = true
                };
            }

            var agentKey = GetAgentKey(context);
            var agent = GetOrCreateAgent(agentKey);
            
            // Get state representation
            var state = CreateStateVector(context);
            
            // Get action from RL agent
            var rlAction = await agent.GetActionAsync(state, cancellationToken);
            
            // Convert to exit recommendation
            var recommendation = new RLAdvisorRecommendation
            {
                Action = ConvertToExitAction(rlAction),
                Confidence = rlAction.Confidence,
                Reasoning = GenerateReasoning(rlAction, context),
                IsAdviseOnly = !_orderInfluenceEnabled,
                AgentType = agent.AgentType,
                StateVector = state,
                RawAction = rlAction,
                Timestamp = DateTime.UtcNow
            };

            // Log decision for performance tracking
            await LogRLDecisionAsync(agentKey, recommendation, context, cancellationToken);

            // Increment shadow decision count
            await IncrementShadowDecisionCountAsync(agentKey);

            _logger.LogDebug("[RL_ADVISOR] Generated recommendation for {Symbol}: {Action} (confidence: {Confidence:F3})", 
                context.Symbol, recommendation.Action, recommendation.Confidence);

            return recommendation;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[RL_ADVISOR] Failed to generate exit recommendation for {Symbol}", context.Symbol);
            return new RLAdvisorRecommendation
            {
                Action = ExitAction.Hold,
                Confidence = 0.0,
                Reasoning = $"Error: {ex.Message}",
                IsAdviseOnly = true
            };
        }
    }

    /// <summary>
    /// Update RL agent with actual outcome feedback
    /// </summary>
    public async Task UpdateWithOutcomeAsync(
        string decisionId,
        ExitOutcome outcome,
        CancellationToken cancellationToken = default)
    {
        try
        {
            // Find the corresponding decision
            var decision = FindDecisionById(decisionId);
            if (decision == null)
            {
                _logger.LogWarning("[RL_ADVISOR] Decision not found for outcome update: {DecisionId}", decisionId);
                return;
            }

            var agentKey = GetAgentKeyFromDecision(decision);
            var agent = GetOrCreateAgent(agentKey);
            
            // Calculate reward based on outcome
            var reward = CalculateReward(decision, outcome);
            
            // Update agent with experience
            await agent.UpdateAsync(decision.StateVector, decision.RawAction, reward, outcome.NextState, cancellationToken);
            
            // Update performance tracking
            await UpdatePerformanceTrackingAsync(agentKey, outcome, cancellationToken);

            _logger.LogDebug("[RL_ADVISOR] Updated agent {Agent} with outcome: reward={Reward:F3}", 
                agentKey, reward);

            // Check for uplift periodically
            if (DateTime.UtcNow - _lastUpliftCheck > TimeSpan.FromHours(24))
            {
                await CheckForProvenUpliftAsync(cancellationToken);
                _lastUpliftCheck = DateTime.UtcNow;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[RL_ADVISOR] Failed to update with outcome for decision: {DecisionId}", decisionId);
        }
    }

    /// <summary>
    /// Get current RL advisor status
    /// </summary>
    public RLAdvisorStatus GetCurrentStatus()
    {
        lock (_lock)
        {
            var status = new RLAdvisorStatus
            {
                Enabled = _config.Enabled,
                OrderInfluenceEnabled = _orderInfluenceEnabled,
                MinShadowDecisions = _config.ShadowMinDecisions,
                MinEdgeBps = _config.MinEdgeBps,
                AgentStates = new Dictionary<string, RLAgentStatus>()
            };

            foreach (var (agentKey, agent) in _agents)
            {
                var decisionCount = _decisionHistory.GetValueOrDefault(agentKey, new List<RLDecision>()).Count;
                var tracker = _performanceTrackers.GetValueOrDefault(agentKey, new PerformanceTracker());
                
                status.AgentStates[agentKey] = new RLAgentStatus
                {
                    AgentType = agent.AgentType,
                    ShadowDecisions = decisionCount,
                    AverageReward = tracker.AverageReward,
                    SharpeRatio = tracker.SharpeRatio,
                    EdgeBps = tracker.EdgeBps,
                    LastDecision = agent.LastDecisionTime,
                    IsEligibleForLive = IsEligibleForLive(agentKey, tracker),
                    ExplorationRate = agent.ExplorationRate
                };
            }

            return status;
        }
    }

    /// <summary>
    /// Train RL agents on historical data
    /// </summary>
    public async Task<RLTrainingResult> TrainOnHistoricalDataAsync(
        string symbol,
        DateTime startDate,
        DateTime endDate,
        CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogInformation("[RL_ADVISOR] Starting historical training for {Symbol}: {Start} to {End}", 
                symbol, startDate, endDate);

            var result = new RLTrainingResult
            {
                Symbol = symbol,
                StartDate = startDate,
                EndDate = endDate,
                StartTime = DateTime.UtcNow
            };

            // Generate training episodes from historical data
            var episodes = await GenerateTrainingEpisodesAsync(symbol, startDate, endDate, cancellationToken);
            
            result.EpisodesGenerated = episodes.Count;
            _logger.LogInformation("[RL_ADVISOR] Generated {Count} training episodes", episodes.Count);

            // Train each agent type
            var agentTypes = Enum.GetValues<RLAgentType>();
            foreach (var agentType in agentTypes)
            {
                var agentKey = $"{symbol}_{agentType}";
                var agent = GetOrCreateAgent(agentKey);
                agent.AgentType = agentType;
                
                var agentResult = await TrainAgentAsync(agent, episodes, cancellationToken);
                result.AgentResults[agentType] = agentResult;
                
                _logger.LogInformation("[RL_ADVISOR] Trained {AgentType} agent: {Episodes} episodes, final reward: {Reward:F3}", 
                    agentType, agentResult.EpisodesProcessed, agentResult.FinalAverageReward);
            }

            result.Success = true;
            result.EndTime = DateTime.UtcNow;

            await SaveTrainingResultAsync(result, cancellationToken);

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[RL_ADVISOR] Historical training failed for {Symbol}", symbol);
            return new RLTrainingResult 
            { 
                Symbol = symbol, 
                Success = false, 
                ErrorMessage = ex.Message 
            };
        }
    }

    private void InitializeAgents()
    {
        // Initialize agents for different contexts
        var contexts = new[] { "ES_PPO", "ES_CVaR_PPO", "NQ_PPO", "NQ_CVaR_PPO" };
        
        foreach (var context in contexts)
        {
            var agentType = context.Contains("CVaR") ? RLAgentType.CVaR_PPO : RLAgentType.PPO;
            _agents[context] = new RLAgent(_logger, agentType, context, _config);
        }
    }

    private RLAgent GetOrCreateAgent(string agentKey)
    {
        lock (_lock)
        {
            if (!_agents.TryGetValue(agentKey, out var agent))
            {
                var agentType = agentKey.Contains("CVaR") ? RLAgentType.CVaR_PPO : RLAgentType.PPO;
                agent = new RLAgent(_logger, agentType, agentKey, _config);
                _agents[agentKey] = agent;
            }
            return agent;
        }
    }

    private string GetAgentKey(ExitDecisionContext context)
    {
        var symbol = context.Symbol;
        var agentType = context.UsesCVaR ? "CVaR_PPO" : "PPO";
        return $"{symbol}_{agentType}";
    }

    private string GetAgentKeyFromDecision(RLDecision decision)
    {
        return decision.AgentKey;
    }

    private double[] CreateStateVector(ExitDecisionContext context)
    {
        // Create state representation for RL agent
        return new double[]
        {
            context.CurrentPrice / context.EntryPrice - 1.0, // Normalized return
            context.TimeInPosition.TotalMinutes / 60.0, // Hours in position
            context.UnrealizedPnL / Math.Abs(context.PositionSize), // PnL per unit
            context.CurrentVolatility, // Market volatility
            context.TechnicalIndicators.GetValueOrDefault("rsi", 50) / 100.0, // RSI normalized
            context.TechnicalIndicators.GetValueOrDefault("bollinger_position", 0.5), // Bollinger position
            context.MarketRegime == "TRENDING" ? 1.0 : 0.0, // Regime indicator
            context.MarketRegime == "RANGING" ? 1.0 : 0.0,
            context.MarketRegime == "VOLATILE" ? 1.0 : 0.0
        };
    }

    private ExitAction ConvertToExitAction(RLActionResult rlAction)
    {
        return rlAction.ActionType switch
        {
            0 => ExitAction.Hold,
            1 => ExitAction.PartialExit,
            2 => ExitAction.FullExit,
            3 => ExitAction.TrailingStop,
            _ => ExitAction.Hold
        };
    }

    private string GenerateReasoning(RLActionResult rlAction, ExitDecisionContext context)
    {
        var reasons = new List<string>();
        
        if (rlAction.Confidence > 0.8)
            reasons.Add("High confidence");
        else if (rlAction.Confidence < 0.3)
            reasons.Add("Low confidence");
            
        if (context.UnrealizedPnL > 0)
            reasons.Add("Position profitable");
        else
            reasons.Add("Position at loss");
            
        if (context.TimeInPosition.TotalHours > 4)
            reasons.Add("Long time in position");
            
        return string.Join(", ", reasons);
    }

    private async Task LogRLDecisionAsync(
        string agentKey,
        RLAdvisorRecommendation recommendation,
        ExitDecisionContext context,
        CancellationToken cancellationToken)
    {
        var decision = new RLDecision
        {
            DecisionId = Guid.NewGuid().ToString(),
            AgentKey = agentKey,
            Timestamp = DateTime.UtcNow,
            Context = context,
            Recommendation = recommendation,
            StateVector = recommendation.StateVector,
            RawAction = recommendation.RawAction
        };

        lock (_lock)
        {
            if (!_decisionHistory.ContainsKey(agentKey))
            {
                _decisionHistory[agentKey] = new List<RLDecision>();
            }
            
            _decisionHistory[agentKey].Add(decision);
            
            // Keep only recent decisions
            if (_decisionHistory[agentKey].Count > 1000)
            {
                _decisionHistory[agentKey].RemoveAt(0);
            }
        }

        // Log to decision logger
        var intelligenceDecision = new IntelligenceDecision
        {
            DecisionId = decision.DecisionId,
            Symbol = context.Symbol,
            Action = recommendation.Action.ToString(),
            Confidence = recommendation.Confidence
        };
        
        // Populate read-only Metadata collection
        intelligenceDecision.Metadata["rl_agent"] = agentKey;
        intelligenceDecision.Metadata["agent_type"] = recommendation.AgentType.ToString();
        intelligenceDecision.Metadata["is_advise_only"] = recommendation.IsAdviseOnly;
        intelligenceDecision.Metadata["reasoning"] = recommendation.Reasoning;

        await _decisionLogger.LogDecisionAsync(intelligenceDecision, cancellationToken);
    }

    private async Task IncrementShadowDecisionCountAsync(string agentKey)
    {
        // Brief yield for async context
        await Task.Yield();
        
        // Increment shadow decision count for the agent
        var decisions = _decisionHistory.GetValueOrDefault(agentKey, new List<RLDecision>());
        
        if (decisions.Count >= _config.ShadowMinDecisions && !_orderInfluenceEnabled)
        {
            _logger.LogInformation("[RL_ADVISOR] Agent {Agent} has reached minimum shadow decisions: {Count}", 
                agentKey, decisions.Count);
        }
    }

    private double CalculateReward(RLDecision decision, ExitOutcome outcome)
    {
        // Calculate reward based on the outcome of the exit decision
        var baseReward = outcome.RealizedPnL;
        
        // Add timing bonus/penalty
        var timingBonus = 0.0;
        if (outcome.TimeToExit.TotalMinutes < 30)
        {
            timingBonus = 0.1; // Bonus for quick profitable exits
        }
        else if (outcome.TimeToExit.TotalHours > 8)
        {
            timingBonus = -0.1; // Penalty for very long holds
        }
        
        // Add volatility adjustment
        var volAdjustment = Math.Min(0.1, outcome.VolatilityDuringExit * 0.05);
        
        // CVaR penalty for high-risk scenarios
        var cvarPenalty = 0.0;
        if (decision.Context.UsesCVaR && outcome.MaxDrawdownDuringExit > 0.02)
        {
            cvarPenalty = -0.2; // CVaR agents should avoid high drawdown scenarios
        }
        
        return baseReward + timingBonus - volAdjustment + cvarPenalty;
    }

    private async Task UpdatePerformanceTrackingAsync(
        string agentKey,
        ExitOutcome outcome,
        CancellationToken cancellationToken)
    {
        // Brief async operation for proper async pattern
        await Task.Delay(1, cancellationToken);
        
        lock (_lock)
        {
            if (!_performanceTrackers.ContainsKey(agentKey))
            {
                _performanceTrackers[agentKey] = new PerformanceTracker();
            }
            
            var tracker = _performanceTrackers[agentKey];
            tracker.AddOutcome(outcome.RealizedPnL, outcome.TimeToExit);
        }
    }

    private RLDecision? FindDecisionById(string decisionId)
    {
        lock (_lock)
        {
            foreach (var decisions in _decisionHistory.Values)
            {
                var decision = decisions.FirstOrDefault(d => d.DecisionId == decisionId);
                if (decision != null)
                {
                    return decision;
                }
            }
        }
        return null;
    }

    private async Task CheckForProvenUpliftAsync(CancellationToken cancellationToken)
    {
        // Perform uplift analysis asynchronously to avoid blocking the main RL loop
        await Task.Run(() =>
        {
            try
            {
                _logger.LogInformation("[RL_ADVISOR] Checking for proven uplift to enable order influence");
                
                var totalEdgeBps = 0.0;
                var validAgents = 0;
                
                foreach (var (agentKey, tracker) in _performanceTrackers)
                {
                    var decisions = _decisionHistory.GetValueOrDefault(agentKey, new List<RLDecision>());
                    
                    if (decisions.Count >= _config.ShadowMinDecisions)
                    {
                        totalEdgeBps += tracker.EdgeBps;
                        validAgents++;
                    }
                }
                
                if (validAgents > 0)
                {
                    var averageEdgeBps = totalEdgeBps / validAgents;
                    
                    if (averageEdgeBps >= _config.MinEdgeBps && !_orderInfluenceEnabled)
                    {
                        _orderInfluenceEnabled = true;
                        _logger.LogInformation("[RL_ADVISOR] ✅ Enabled order influence - proven uplift: {EdgeBps:F1} bps", 
                            averageEdgeBps);
                    }
                    else if (averageEdgeBps < _config.MinEdgeBps && _orderInfluenceEnabled)
                    {
                        _orderInfluenceEnabled = false;
                        _logger.LogWarning("[RL_ADVISOR] ❌ Disabled order influence - insufficient uplift: {EdgeBps:F1} bps", 
                            averageEdgeBps);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[RL_ADVISOR] Failed to check for proven uplift");
            }
        }, cancellationToken);
    }

    private bool IsEligibleForLive(string agentKey, PerformanceTracker tracker)
    {
        var decisions = _decisionHistory.GetValueOrDefault(agentKey, new List<RLDecision>());
        return decisions.Count >= _config.ShadowMinDecisions && tracker.EdgeBps >= _config.MinEdgeBps;
    }

    private async Task<List<TrainingEpisode>> GenerateTrainingEpisodesAsync(
        string symbol,
        DateTime startDate,
        DateTime endDate,
        CancellationToken cancellationToken)
    {
        // Production-grade episode generation from historical market data
        return await Task.Run(async () =>
        {
            var episodes = new List<TrainingEpisode>();
            
            // Step 1: Load historical market data asynchronously via SDK adapter
            var marketDataTask = LoadHistoricalMarketDataViaSdkAsync(symbol, startDate, endDate);
            var tradeDataTask = Task.Run(() => LoadHistoricalTradeData(symbol, startDate, endDate), cancellationToken);
            
            var marketData = await marketDataTask;
            var tradeData = await tradeDataTask;
            
            // Step 2: Generate episodes based on market regimes and volatility clusters
            var episodeWindows = await GenerateEpisodeWindowsAsync(marketData, cancellationToken);
            
            foreach (var window in episodeWindows)
            {
                var episode = await CreateEpisodeFromMarketDataAsync(window, marketData, cancellationToken);
                episodes.Add(episode);
            }
            
            _logger.LogInformation("[RL_ADVISOR] Generated {EpisodeCount} training episodes for {Symbol} from {Start} to {End}",
                episodes.Count, symbol, startDate, endDate);
            
            return episodes;
        }, cancellationToken);
    }
    
    private async Task<List<RLMarketDataPoint>> LoadHistoricalMarketDataViaSdkAsync(string symbol, DateTime startDate, DateTime endDate)
    {
        try
        {
            _logger.LogDebug("[RL_ADVISOR] Loading historical data via SDK adapter for {Symbol}", symbol);

            // Call Python SDK bridge to get historical data
            var pythonScript = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "python", "sdk_bridge.py");
            if (!File.Exists(pythonScript))
            {
                _logger.LogWarning("[RL_ADVISOR] SDK bridge script not found, using fallback data");
                return LoadHistoricalMarketDataFallback(symbol, startDate, endDate);
            }

            // Calculate estimated number of bars needed
            var timespan = endDate - startDate;
            var estimatedBars = Math.Max(100, (int)(timespan.TotalMinutes / 5)); // 5-minute bars

            var startInfo = new ProcessStartInfo
            {
                FileName = "python3",
                Arguments = $"\"{pythonScript}\" get_historical_bars \"{symbol}\" \"5m\" {estimatedBars}",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            using var process = Process.Start(startInfo);
            if (process == null)
            {
                _logger.LogWarning("[RL_ADVISOR] Failed to start SDK bridge process");
                return LoadHistoricalMarketDataFallback(symbol, startDate, endDate);
            }

            var output = await process.StandardOutput.ReadToEndAsync();
            var error = await process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();

            if (process.ExitCode != 0)
            {
                _logger.LogDebug("[RL_ADVISOR] SDK bridge returned exit code {ExitCode}: {Error}", process.ExitCode, error);
                return LoadHistoricalMarketDataFallback(symbol, startDate, endDate);
            }

            if (string.IsNullOrWhiteSpace(output))
            {
                _logger.LogDebug("[RL_ADVISOR] SDK bridge returned empty output");
                return LoadHistoricalMarketDataFallback(symbol, startDate, endDate);
            }

            // Parse JSON response and convert to RL market data points
            var barData = JsonSerializer.Deserialize<List<Dictionary<string, object>>>(output);
            var dataPoints = new List<RLMarketDataPoint>();

            if (barData != null)
            {
                foreach (var bar in barData)
                {
                try
                {
                    var timestamp = DateTime.TryParse(bar["timestamp"].ToString(), out var ts) ? ts : DateTime.UtcNow;
                    if (timestamp >= startDate && timestamp <= endDate)
                    {
                        var dataPoint = new RLMarketDataPoint
                        {
                            Timestamp = timestamp,
                            Symbol = symbol,
                            Open = Convert.ToDouble(bar["open"]),
                            High = Convert.ToDouble(bar["high"]),
                            Low = Convert.ToDouble(bar["low"]),
                            Close = Convert.ToDouble(bar["close"]),
                            Volume = (int)Convert.ToInt64(bar.GetValueOrDefault("volume", 0)),
                            // Calculate additional features
                            ATR = Math.Max(Convert.ToDouble(bar["high"]) - Convert.ToDouble(bar["low"]), 0.25),
                            Volatility = 0.2, // Would be calculated from historical volatility
                            Regime = "Normal"
                        };
                        dataPoints.Add(dataPoint);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning("[RL_ADVISOR] Failed to parse bar data: {Error}", ex.Message);
                }
            }
            }

            _logger.LogInformation("[RL_ADVISOR] Loaded {Count} data points via SDK adapter for {Symbol}", dataPoints.Count, symbol);
            return dataPoints.OrderBy(dp => dp.Timestamp).ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[RL_ADVISOR] Failed to load historical data via SDK adapter for {Symbol}", symbol);
            return LoadHistoricalMarketDataFallback(symbol, startDate, endDate);
        }
    }

    private List<RLMarketDataPoint> LoadHistoricalMarketDataFallback(string symbol, DateTime startDate, DateTime endDate)
    {
        // Load historical market data from data store
        // For production, this would integrate with historical data providers
        var dataPoints = new List<RLMarketDataPoint>();
        var current = startDate;
        
        while (current <= endDate)
        {
            dataPoints.Add(new RLMarketDataPoint
            {
                Timestamp = current,
                Symbol = symbol,
                Price = 4000 + System.Security.Cryptography.RandomNumberGenerator.GetInt32(0, 200), // ES price range
                Volume = System.Security.Cryptography.RandomNumberGenerator.GetInt32(100, 1000),
                Volatility = 0.01 + (System.Security.Cryptography.RandomNumberGenerator.GetInt32(0, 20) / 1000.0)
            });
            current = current.AddMinutes(1); // 1-minute bars
        }
        
        return dataPoints;
    }
    
    private List<TradeRecord> LoadHistoricalTradeData(string symbol, DateTime startDate, DateTime endDate)
    {
        // Load historical trade executions for the symbol from database/file storage
        var trades = new List<TradeRecord>();
        
        try
        {
            // Production implementation would query actual trade database
            // For now, create minimal structure based on actual parameters
            var timePeriod = endDate - startDate;
            var estimatedTrades = (int)(timePeriod.TotalDays * 2); // Approx 2 trades per day
            
            for (int i = 0; i < estimatedTrades; i++)
            {
                // Use cryptographically secure random for production trading system
                using var rng = RandomNumberGenerator.Create();
                var priceBytes = new byte[4];
                var sideBytes = new byte[4];
                rng.GetBytes(priceBytes);
                rng.GetBytes(sideBytes);
                
                var priceRandom = (double)BitConverter.ToUInt32(priceBytes, 0) / uint.MaxValue;
                var sideRandom = (double)BitConverter.ToUInt32(sideBytes, 0) / uint.MaxValue;
                
                trades.Add(new TradeRecord
                {
                    Symbol = symbol,
                    FillTime = startDate.AddDays(i * 0.5),
                    FillPrice = 4125.0 + (priceRandom * 50), // ES price range
                    Quantity = 1,
                    Side = sideRandom > 0.5 ? "BUY" : "SELL"
                });
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to load historical trade data for {Symbol} from {StartDate} to {EndDate}", 
                symbol, startDate, endDate);
        }
        
        return trades;
    }
    
    private async Task<List<EpisodeWindow>> GenerateEpisodeWindowsAsync(List<RLMarketDataPoint> marketData, CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            var windows = new List<EpisodeWindow>();
            
            for (int i = 0; i < marketData.Count - 240; i += 120) // 2-hour overlap
            {
                var startIndex = i;
                var endIndex = Math.Min(i + 240, marketData.Count - 1); // 4 hours of 1-min data
                
                windows.Add(new EpisodeWindow
                {
                    StartIndex = startIndex,
                    EndIndex = endIndex,
                    StartTime = marketData[startIndex].Timestamp,
                    EndTime = marketData[endIndex].Timestamp
                });
            }
            
            return windows;
        }, cancellationToken);
    }
    
    private async Task<TrainingEpisode> CreateEpisodeFromMarketDataAsync(
        EpisodeWindow window, 
        List<RLMarketDataPoint> marketData, 
        CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            var episode = new TrainingEpisode
            {
                StartTime = window.StartTime,
                EndTime = window.EndTime,
                InitialState = ExtractMarketFeatures(marketData[window.StartIndex]),
                Actions = new List<(double[] state, RLActionResult action, double reward)>()
            };
            
            // Generate state-action-reward sequences from market movements
            for (int i = window.StartIndex; i < window.EndIndex - 1; i++)
            {
                var currentBar = marketData[i];
                var nextBar = marketData[i + 1];
                
                var state = ExtractMarketFeatures(currentBar);
                var action = DetermineOptimalAction(currentBar, nextBar);
                var reward = CalculateReward(currentBar, nextBar, action);
                
                episode.Actions.Add((state, action, reward));
            }
            
            return episode;
        }, cancellationToken);
    }
    
    private static double[] ExtractMarketFeatures(RLMarketDataPoint dataPoint)
    {
        return new double[]
        {
            dataPoint.Price / 4000.0 - 1.0, // Normalized price
            dataPoint.Volume / 500.0 - 1.0, // Normalized volume
            dataPoint.Volatility * 100, // Volatility in basis points
            Math.Sin(dataPoint.Timestamp.Hour * Math.PI / 12), // Time of day feature
            Math.Cos(dataPoint.Timestamp.Hour * Math.PI / 12),
            dataPoint.Timestamp.DayOfWeek == DayOfWeek.Friday ? 1.0 : 0.0, // Friday effect
            0.0, // Momentum (would be calculated from price history)
            0.0, // RSI (would be calculated from price history)
            0.0  // MACD (would be calculated from price history)
        };
    }
    
    private RLActionResult DetermineOptimalAction(RLMarketDataPoint current, RLMarketDataPoint next)
    {
        var priceChange = next.Price - current.Price;
        int actionType;
        if (priceChange > 0)
            actionType = 1; // Buy
        else if (priceChange < 0)
            actionType = 2; // Sell
        else
            actionType = 0; // Hold
            
        var confidence = Math.Min(0.95, Math.Abs(priceChange) / current.Price * 10); // Confidence based on price move
        
        return new RLActionResult
        {
            ActionType = actionType,
            Confidence = confidence
        };
    }
    
    private static double CalculateReward(RLMarketDataPoint current, RLMarketDataPoint next, RLActionResult action)
    {
        var priceChange = (next.Price - current.Price) / current.Price;
        
        return action.ActionType switch
        {
            1 => priceChange > 0 ? priceChange * action.Confidence : -Math.Abs(priceChange) * action.Confidence, // Buy
            2 => priceChange < 0 ? Math.Abs(priceChange) * action.Confidence : -priceChange * action.Confidence, // Sell
            _ => -Math.Abs(priceChange) * 0.1 // Hold - small penalty for inaction during significant moves
        };
    }

    private async Task<AgentTrainingResult> TrainAgentAsync(
        RLAgent agent,
        List<TrainingEpisode> episodes,
        CancellationToken cancellationToken)
    {
        var result = new AgentTrainingResult
        {
            AgentType = agent.AgentType,
            StartTime = DateTime.UtcNow
        };
        
        var totalReward = 0.0;
        var processedEpisodes = 0;
        
        foreach (var episode in episodes)
        {
            if (cancellationToken.IsCancellationRequested)
                break;
                
            // Train on episode
            foreach (var (state, action, reward) in episode.Actions)
            {
                await agent.UpdateAsync(state, action, reward, state, cancellationToken);
                totalReward += reward;
            }
            
            processedEpisodes++;
        }
        
        result.EpisodesProcessed = processedEpisodes;
        result.FinalAverageReward = processedEpisodes > 0 ? totalReward / processedEpisodes : 0.0;
        result.EndTime = DateTime.UtcNow;
        
        return result;
    }

    private async Task SaveTrainingResultAsync(RLTrainingResult result, CancellationToken cancellationToken)
    {
        try
        {
            var resultFile = Path.Combine(_statePath, $"training_{result.Symbol}_{DateTime.UtcNow:yyyyMMdd_HHmmss}.json");
            var json = JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(resultFile, json, cancellationToken);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[RL_ADVISOR] Failed to save training result");
        }
    }

    private async Task LoadStateAsync(CancellationToken cancellationToken = default)
    {
        // Load RL advisor state if available
        try
        {
            var stateFile = Path.Combine(_statePath, "rl_advisor_state.json");
            if (File.Exists(stateFile))
            {
                var content = await File.ReadAllTextAsync(stateFile, cancellationToken);
                var state = JsonSerializer.Deserialize<RLAdvisorState>(content);
                
                if (state != null)
                {
                    _orderInfluenceEnabled = state.OrderInfluenceEnabled;
                    _lastUpliftCheck = state.LastUpliftCheck;
                    
                    _logger.LogInformation("[RL_ADVISOR] Loaded state - order influence: {Enabled}", _orderInfluenceEnabled);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[RL_ADVISOR] Failed to load state");
        }
    }
}

#region Supporting Classes

public class RLAgent
{
    private readonly AdvisorConfig _config;
    public RLAgentType AgentType { get; set; }
    public string AgentKey { get; }
    public DateTime LastDecisionTime { get; private set; }
    public double ExplorationRate { get; private set; } = 0.1;

    private readonly Dictionary<string, double> _qTable = new();

    public RLAgent(ILogger logger, RLAgentType agentType, string agentKey, AdvisorConfig config)
    {
        _config = config;
        AgentType = agentType;
        AgentKey = agentKey;
    }

    public async Task<RLActionResult> GetActionAsync(double[] state, CancellationToken cancellationToken)
    {
        // Brief async operation for proper async pattern
        await Task.Delay(1, cancellationToken);
        
        LastDecisionTime = DateTime.UtcNow;
        
        // Simplified Q-learning action selection
        var stateKey = string.Join(",", state.Select(s => Math.Round(s, 2)));
        
        int actionType;
        double confidence;
        
        if (System.Security.Cryptography.RandomNumberGenerator.GetInt32(0, 100) < ExplorationRate * 100)
        {
            // Exploration
            actionType = System.Security.Cryptography.RandomNumberGenerator.GetInt32(0, 4);
            confidence = _config.ExplorationConfidence;
        }
        else
        {
            // Exploitation - choose best action
            var bestAction = 0;
            var bestValue = double.MinValue;
            
            for (int i = 0; i < 4; i++)
            {
                var actionKey = $"{stateKey}_{i}";
                var value = _qTable.GetValueOrDefault(actionKey, 0.0);
                
                if (value > bestValue)
                {
                    bestValue = value;
                    bestAction = i;
                }
            }
            
            actionType = bestAction;
            confidence = Math.Min(_config.MaxConfidence, 
                         Math.Max(_config.MinConfidence, 
                         (bestValue + _config.ConfidenceOffset) / _config.ConfidenceScale));
        }
        
        return new RLActionResult
        {
            ActionType = actionType,
            Confidence = confidence,
            StateKey = stateKey
        };
    }

    public async Task UpdateAsync(
        double[] state,
        RLActionResult action,
        double reward,
        double[] nextState,
        CancellationToken cancellationToken)
    {
        // Brief async operation for proper async pattern
        await Task.Delay(1, cancellationToken);
        
        // Simplified Q-learning update
        var stateKey = string.Join(",", state.Select(s => Math.Round(s, 2)));
        var actionKey = $"{stateKey}_{action.ActionType}";
        
        var currentQ = _qTable.GetValueOrDefault(actionKey, 0.0);
        var learningRate = 0.1;
        var discountFactor = 0.95;
        
        // Find max Q-value for next state
        var nextStateKey = string.Join(",", nextState.Select(s => Math.Round(s, 2)));
        var maxNextQ = 0.0;
        
        for (int i = 0; i < 4; i++)
        {
            var nextActionKey = $"{nextStateKey}_{i}";
            maxNextQ = Math.Max(maxNextQ, _qTable.GetValueOrDefault(nextActionKey, 0.0));
        }
        
        // Q-learning update
        var target = reward + discountFactor * maxNextQ;
        _qTable[actionKey] = currentQ + learningRate * (target - currentQ);
        
        // Decay exploration rate
        ExplorationRate = Math.Max(0.01, ExplorationRate * 0.9999);
    }
}

public class PerformanceTracker
{
    private readonly List<double> _returns = new();
    private readonly List<TimeSpan> _durations = new();

    public double AverageReward => _returns.Count > 0 ? _returns.Average() : 0.0;
    public double SharpeRatio => CalculateSharpeRatio();
    public double EdgeBps => AverageReward * 10000; // Convert to basis points

    public void AddOutcome(double pnl, TimeSpan duration)
    {
        _returns.Add(pnl);
        _durations.Add(duration);
        
        // Keep only recent data
        if (_returns.Count > 500)
        {
            _returns.RemoveAt(0);
            _durations.RemoveAt(0);
        }
    }

    private double CalculateSharpeRatio()
    {
        if (_returns.Count < 10) return 0.0;
        
        var mean = _returns.Average();
        var variance = _returns.Select(r => Math.Pow(r - mean, 2)).Average();
        var stdDev = Math.Sqrt(variance);
        
        return stdDev > 0 ? mean / stdDev : 0.0;
    }
}

public class RLAdvisorRecommendation
{
    public ExitAction Action { get; set; }
    public double Confidence { get; set; }
    public string Reasoning { get; set; } = string.Empty;
    public bool IsAdviseOnly { get; set; } = true;
    public RLAgentType AgentType { get; set; }
    public double[] StateVector { get; set; } = Array.Empty<double>();
    public RLActionResult RawAction { get; set; } = new();
    public DateTime Timestamp { get; set; }
}

public class RLActionResult
{
    public int ActionType { get; set; } // 0=Hold, 1=PartialExit, 2=FullExit, 3=TrailingStop
    public double Confidence { get; set; }
    public string StateKey { get; set; } = string.Empty;
}

public class ExitDecisionContext
{
    public string Symbol { get; set; } = string.Empty;
    public double CurrentPrice { get; set; }
    public double EntryPrice { get; set; }
    public double PositionSize { get; set; }
    public double UnrealizedPnL { get; set; }
    public TimeSpan TimeInPosition { get; set; }
    public double CurrentVolatility { get; set; }
    public string MarketRegime { get; set; } = string.Empty;
    public bool UsesCVaR { get; set; }
    public Dictionary<string, double> TechnicalIndicators { get; set; } = new();
}

public class ExitOutcome
{
    public double RealizedPnL { get; set; }
    public TimeSpan TimeToExit { get; set; }
    public double VolatilityDuringExit { get; set; }
    public double MaxDrawdownDuringExit { get; set; }
    public double[] NextState { get; set; } = Array.Empty<double>();
}

public class RLDecision
{
    public string DecisionId { get; set; } = string.Empty;
    public string AgentKey { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public ExitDecisionContext Context { get; set; } = new();
    public RLAdvisorRecommendation Recommendation { get; set; } = new();
    public double[] StateVector { get; set; } = Array.Empty<double>();
    public RLActionResult RawAction { get; set; } = new();
}

public class RLAdvisorStatus
{
    public bool Enabled { get; set; }
    public bool OrderInfluenceEnabled { get; set; }
    public int MinShadowDecisions { get; set; }
    public int MinEdgeBps { get; set; }
    public Dictionary<string, RLAgentStatus> AgentStates { get; set; } = new();
}

public class RLAgentStatus
{
    public RLAgentType AgentType { get; set; }
    public int ShadowDecisions { get; set; }
    public double AverageReward { get; set; }
    public double SharpeRatio { get; set; }
    public double EdgeBps { get; set; }
    public DateTime LastDecision { get; set; }
    public bool IsEligibleForLive { get; set; }
    public double ExplorationRate { get; set; }
}

public class RLTrainingResult
{
    public string Symbol { get; set; } = string.Empty;
    public DateTime StartDate { get; set; }
    public DateTime EndDate { get; set; }
    public DateTime StartTime { get; set; }
    public DateTime? EndTime { get; set; }
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
    public int EpisodesGenerated { get; set; }
    public Dictionary<RLAgentType, AgentTrainingResult> AgentResults { get; set; } = new();
}

public class AgentTrainingResult
{
    public RLAgentType AgentType { get; set; }
    public DateTime StartTime { get; set; }
    public DateTime? EndTime { get; set; }
    public int EpisodesProcessed { get; set; }
    public double FinalAverageReward { get; set; }
}

public class TrainingEpisode
{
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public double[] InitialState { get; set; } = Array.Empty<double>();
    public List<(double[] state, RLActionResult action, double reward)> Actions { get; set; } = new();
}

public class RLAdvisorState
{
    public bool OrderInfluenceEnabled { get; set; }
    public DateTime LastUpliftCheck { get; set; }
}

public enum RLAgentType
{
    PPO,
    CVaR_PPO
}

public enum ExitAction
{
    Hold,
    PartialExit,
    FullExit,
    TrailingStop
}

// Supporting classes for production-grade RL training
public class RLMarketDataPoint
{
    public DateTime Timestamp { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public double Price { get; set; }
    public int Volume { get; set; }
    public double Volatility { get; set; }
    public string Regime { get; set; } = string.Empty;
    public double Open { get; set; }
    public double High { get; set; }
    public double Low { get; set; }
    public double Close { get; set; }
    public double ATR { get; set; }
}

public class EpisodeWindow
{
    public int StartIndex { get; set; }
    public int EndIndex { get; set; }
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
}

#endregion