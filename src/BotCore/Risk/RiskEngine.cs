using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Diagnostics;

namespace BotCore.Risk
{
    public sealed class RiskEngine
    {
        public RiskConfig cfg { get; set; } = new RiskConfig();
        private readonly DrawdownProtectionSystem _drawdownProtection;

        public RiskEngine()
        {
            _drawdownProtection = new DrawdownProtectionSystem();
        }

        public Task InitializeAsync(decimal startingBalance)
        {
            return _drawdownProtection.InitializeDrawdownProtection(startingBalance);
        }

        public Task UpdateBalanceAsync(decimal currentBalance, Trade? lastTrade = null)
        {
            return _drawdownProtection.UpdateBalance(currentBalance, lastTrade);
        }

        public static decimal ComputeRisk(decimal entry, decimal stop, decimal target, bool isLong)
        {
            var risk = isLong ? entry - stop : stop - entry;
            var reward = isLong ? target - entry : entry - target;
            if (risk <= 0 || reward < 0) return 0m;
            return reward / risk;
        }

        public static decimal size_for(decimal riskPerTrade, decimal dist, decimal pointValue)
        {
            if (dist <= 0 || pointValue <= 0) return 0m;
            return Math.Floor(riskPerTrade / (dist * pointValue));
        }

        // NEW: Equity-% aware sizing helper (backwards-compatible)
        public (int Qty, decimal UsedRpt) ComputeSize(string symbol, decimal entry, decimal stop, decimal accountEquity)
        {
            var dist = Math.Abs(entry - stop);
            if (dist <= 0) return (0, 0);
            var pv = BotCore.Models.InstrumentMeta.PointValue(symbol);
            if (pv <= 0) return (0, 0);

            // If equity% configured and equity provided, use it, else fall back to fixed RPT
            var usePct = cfg.risk_pct_of_equity > 0m && accountEquity > 0m;
            var rpt = usePct ? Math.Round(accountEquity * cfg.risk_pct_of_equity, 2) : cfg.risk_per_trade;
            var raw = (int)System.Math.Floor((double)(rpt / (dist * pv)));
            var lot = BotCore.Models.InstrumentMeta.LotStep(symbol);
            var qty = System.Math.Max(0, raw - (raw % System.Math.Max(1, lot)));
            
            // Apply drawdown protection multiplier
            var sizeMultiplier = _drawdownProtection.GetPositionSizeMultiplier();
            qty = (int)(qty * sizeMultiplier);
            
            return (qty, rpt);
        }

        public bool ShouldHaltDay(decimal realizedPnlToday) => cfg.max_daily_drawdown > 0 && -realizedPnlToday >= cfg.max_daily_drawdown;
        public bool ShouldHaltWeek(decimal realizedPnlWeek) => cfg.max_weekly_drawdown > 0 && -realizedPnlWeek >= cfg.max_weekly_drawdown;

        public DrawdownAnalysis GetDrawdownAnalysis() => _drawdownProtection.AnalyzeDrawdownPattern();
    }

    public class Trade
    {
        public decimal PnL { get; set; }
        public DateTime Timestamp { get; set; }
        public string Symbol { get; set; } = string.Empty;
        public int Quantity { get; set; }
        public decimal Price { get; set; }
    }

    public class DrawdownAnalysis
    {
        public decimal CurrentDrawdown { get; set; }
        public decimal MaxDrawdown { get; set; }
        public double DrawdownPercent { get; set; }
        public int ConsecutiveLosses { get; set; }
        public decimal RecoveryRequired { get; set; }
        public TimeSpan EstimatedRecoveryTime { get; set; }
        public string RiskLevel { get; set; } = string.Empty;
        public string RecommendedAction { get; set; } = string.Empty;
    }

    public class DrawdownAlert
    {
        public string Level { get; set; } = string.Empty;
        public decimal DrawdownAmount { get; set; }
        public double DrawdownPercent { get; set; }
        public string Action { get; set; } = string.Empty;
        public DateTime Timestamp { get; set; }
    }

    // ================================================================================
    // COMPONENT 9: DRAWDOWN CURVE BREAKER
    // ================================================================================

    public class DrawdownProtectionSystem
    {
        private readonly ConcurrentDictionary<string, DrawdownTracker> _trackers = new();
        private readonly Timer _drawdownMonitor;
        private decimal _dailyStartBalance;
        private int _consecutiveLosses;
        private decimal _peakBalance;
        private bool _tradingHalted;
        private decimal _positionSizeMultiplier = 1.0m;
        
        public class DrawdownTracker
        {
            public string TrackerId { get; set; } = string.Empty;
            public decimal PeakValue { get; set; }
            public decimal CurrentValue { get; set; }
            public decimal DrawdownAmount { get; set; }
            public double DrawdownPercent { get; set; }
            public DateTime PeakTime { get; set; }
            public DateTime DrawdownStart { get; set; }
            public TimeSpan DrawdownDuration { get; set; }
            public int ConsecutiveLosses { get; set; }
            public List<decimal> LossSequence { get; } = new();
        }
        
        public class DrawdownAction
        {
            public string ActionType { get; set; } = string.Empty;
            public decimal TriggerLevel { get; set; }
            public string Description { get; set; } = string.Empty;
            public Func<Task> Action { get; set; } = null!;
        }
        
        private readonly List<DrawdownAction> _drawdownActions = new();

        public DrawdownProtectionSystem()
        {
            _drawdownMonitor = new Timer(MonitorDrawdown, null, Timeout.InfiniteTimeSpan, Timeout.InfiniteTimeSpan);
            InitializeDrawdownActions();
        }

        private void InitializeDrawdownActions()
        {
            _drawdownActions.AddRange(new[]
            {
                new DrawdownAction { 
                    ActionType = "REDUCE_SIZE_25", 
                    TriggerLevel = 250m, 
                    Description = "Reduce position size by 25%",
                    Action = async () => await ReducePositionSize(0.75m).ConfigureAwait(false)
                },
                new DrawdownAction { 
                    ActionType = "REDUCE_SIZE_50", 
                    TriggerLevel = 500m, 
                    Description = "Reduce position size by 50%",
                    Action = async () => await ReducePositionSize(0.5m).ConfigureAwait(false)
                },
                new DrawdownAction { 
                    ActionType = "REDUCE_SIZE_75", 
                    TriggerLevel = 750m, 
                    Description = "Reduce position size by 75%",
                    Action = async () => await ReducePositionSize(0.25m).ConfigureAwait(false)
                },
                new DrawdownAction { 
                    ActionType = "STRATEGY_ROTATION", 
                    TriggerLevel = 1000m, 
                    Description = "Switch to conservative strategies only",
                    Action = async () => await SwitchToConservativeMode().ConfigureAwait(false)
                },
                new DrawdownAction { 
                    ActionType = "HALT_NEW_TRADES", 
                    TriggerLevel = 1500m, 
                    Description = "Stop opening new positions",
                    Action = async () => await HaltNewTrades().ConfigureAwait(false)
                },
                new DrawdownAction { 
                    ActionType = "CLOSE_ALL", 
                    TriggerLevel = 2000m, 
                    Description = "Close all positions and halt",
                    Action = async () => await EmergencyCloseAll().ConfigureAwait(false)
                }
            }).ConfigureAwait(false);
        }
        
        public Task InitializeDrawdownProtection(decimal startingBalance)
        {
            _dailyStartBalance = startingBalance;
            _peakBalance = startingBalance;
            
            // Create main tracker
            _trackers["MAIN"] = new DrawdownTracker
            {
                TrackerId = "MAIN",
                PeakValue = startingBalance,
                CurrentValue = startingBalance,
                PeakTime = DateTime.UtcNow
            };
            
            // Start monitoring
            _drawdownMonitor.Change(TimeSpan.Zero, TimeSpan.FromSeconds(1));
            
            return Task.CompletedTask;
        }
        
        public async Task UpdateBalance(decimal currentBalance, Trade? lastTrade = null)
        {
            var tracker = _trackers["MAIN"];
            tracker.CurrentValue = currentBalance;
            
            // Update peak
            if (currentBalance > tracker.PeakValue)
            {
                tracker.PeakValue = currentBalance;
                tracker.PeakTime = DateTime.UtcNow;
                _peakBalance = currentBalance;
                _consecutiveLosses; // Reset on new peak
                _positionSizeMultiplier = 1.0m; // Reset position sizing
            }
            
            // Calculate drawdown
            tracker.DrawdownAmount = tracker.PeakValue - currentBalance;
            tracker.DrawdownPercent = tracker.PeakValue > 0 
                ? (double)(tracker.DrawdownAmount / tracker.PeakValue) * 100 
                : 0;
            
            // Track consecutive losses
            if (lastTrade != null && lastTrade.PnL < 0)
            {
                _consecutiveLosses++;
                tracker.ConsecutiveLosses = _consecutiveLosses;
                tracker.LossSequence.Add(lastTrade.PnL);
                
                // Exponential backoff on losing streak
                await ApplyLosingStreakProtection().ConfigureAwait(false);
            }
            else if (lastTrade != null && lastTrade.PnL > 0)
            {
                _consecutiveLosses;
                tracker.LossSequence.Clear();
            }
            
            // Check drawdown levels and take action
            await CheckDrawdownLevels(tracker).ConfigureAwait(false);
            
            // Psychological threshold check
            await CheckPsychologicalThresholds(currentBalance).ConfigureAwait(false);
        }
        
        private async Task CheckDrawdownLevels(DrawdownTracker tracker)
        {
            foreach (var action in _drawdownActions.OrderBy(a => a.TriggerLevel))
            {
                if (tracker.DrawdownAmount >= action.TriggerLevel)
                {
                    LogCritical($"DRAWDOWN TRIGGER: {action.Description} at ${tracker.DrawdownAmount:F2}");
                    
                    await action.Action().ConfigureAwait(false);
                    
                    // Send alert
                    await SendDrawdownAlert(new DrawdownAlert
                    {
                        Level = action.ActionType,
                        DrawdownAmount = tracker.DrawdownAmount,
                        DrawdownPercent = tracker.DrawdownPercent,
                        Action = action.Description,
                        Timestamp = DateTime.UtcNow
                    }).ConfigureAwait(false);
                    
                    // Don't trigger multiple actions at once
                    break;
                }
            }
        }
        
        private async Task ApplyLosingStreakProtection()
        {
            var protection = _consecutiveLosses switch
            {
                3 => new { Reduction = 0.75m, Cooldown = TimeSpan.FromMinutes(15) },
                4 => new { Reduction = 0.5m, Cooldown = TimeSpan.FromMinutes(30) },
                5 => new { Reduction = 0.25m, Cooldown = TimeSpan.FromHours(1) },
                >= 6 => new { Reduction = 0m, Cooldown = TimeSpan.FromHours(24) },
                _ => null
            };
            
            if (protection != null)
            {
                if (protection.Reduction == 0)
                {
                    await HaltTrading($"Losing streak protection: {_consecutiveLosses} consecutive losses").ConfigureAwait(false);
                }
                else
                {
                    await ReducePositionSize(protection.Reduction).ConfigureAwait(false);
                    await EnforceCooldownPeriod(protection.Cooldown).ConfigureAwait(false);
                }
                
                LogWarning($"Losing streak protection applied: {_consecutiveLosses} losses, {protection.Reduction:P0} size, {protection.Cooldown.TotalMinutes}min cooldown");
            }
        }
        
        private async Task CheckPsychologicalThresholds(decimal currentBalance)
        {
            var dailyPnL = currentBalance - _dailyStartBalance;
            
            // Psychological loss thresholds
            if (dailyPnL <= -1000)
            {
                LogWarning("Psychological threshold: $1000 daily loss");
                
                if (dailyPnL <= -1500)
                {
                    // Take a break
                    await EnforceCooldownPeriod(TimeSpan.FromHours(2)).ConfigureAwait(false);
                    LogCritical("Enforcing 2-hour break after $1500 loss");
                }
            }
            
            // Protect profits
            if (dailyPnL >= 1000)
            {
                // Switch to capital preservation mode
                await EnableProfitProtection(dailyPnL).ConfigureAwait(false);
                
                if (dailyPnL >= 2000)
                {
                    // Consider stopping for the day
                    LogInfo("Excellent day: Consider stopping at $2000 profit");
                }
            }
        }
        
        private void MonitorDrawdown(object? state)
        {
            foreach (var tracker in _trackers.Values)
            {
                if (tracker.DrawdownAmount > 0)
                {
                    tracker.DrawdownDuration = DateTime.UtcNow - tracker.DrawdownStart;
                    
                    // Alert on prolonged drawdown
                    if (tracker.DrawdownDuration > TimeSpan.FromHours(2))
                    {
                        LogWarning($"Prolonged drawdown: {tracker.DrawdownDuration.TotalHours:F1} hours");
                    }
                    
                    // Calculate recovery metrics
                    var recoveryRequired = tracker.DrawdownAmount / (tracker.PeakValue - tracker.DrawdownAmount);
                    if (recoveryRequired > 0.25m) // Need > 25% gain to recover
                    {
                        LogWarning($"Difficult recovery: Need {recoveryRequired:P1} gain to reach peak");
                    }
                }
            }
        }
        
        private Task ReducePositionSize(decimal multiplier)
        {
            _positionSizeMultiplier = multiplier;
            LogAction($"Position size reduced to {multiplier:P0}");
            return Task.CompletedTask;
        }

        public decimal GetPositionSizeMultiplier() => _positionSizeMultiplier;
        
        private Task SwitchToConservativeMode()
        {
            LogAction("Switched to conservative mode");
            return Task.CompletedTask;
        }
        
        private Task HaltNewTrades()
        {
            _tradingHalted = true;
            LogAction("New trades halted due to drawdown");
            return Task.CompletedTask;
        }
        
        private Task EmergencyCloseAll()
        {
            LogCritical("EMERGENCY: Closing all positions due to maximum drawdown");
            return HaltTrading("Maximum drawdown reached");
        }

        private Task HaltTrading(string reason)
        {
            _tradingHalted = true;
            LogCritical($"Trading halted: {reason}");
            return Task.CompletedTask;
        }

        private Task EnforceCooldownPeriod(TimeSpan cooldown)
        {
            LogAction($"Enforcing cooldown period: {cooldown.TotalMinutes} minutes");
            return Task.CompletedTask;
        }

        private Task EnableProfitProtection(decimal profit)
        {
            var protectionLevel = profit * 0.5m; // Protect 50% of profits
            LogInfo($"Profit protection enabled at ${protectionLevel:F2}");
            return Task.CompletedTask;
        }

        private Task SendDrawdownAlert(DrawdownAlert alert)
        {
            LogCritical($"DRAWDOWN ALERT: {alert.Action} - Amount: ${alert.DrawdownAmount:F2} ({alert.DrawdownPercent:F1}%)");
            return Task.CompletedTask;
        }
        
        public DrawdownAnalysis AnalyzeDrawdownPattern()
        {
            var tracker = _trackers["MAIN"];
            
            return new DrawdownAnalysis
            {
                CurrentDrawdown = tracker.DrawdownAmount,
                MaxDrawdown = _trackers.Values.Max(t => t.DrawdownAmount),
                DrawdownPercent = tracker.DrawdownPercent,
                ConsecutiveLosses = tracker.ConsecutiveLosses,
                RecoveryRequired = tracker.DrawdownAmount / (tracker.CurrentValue > 0 ? tracker.CurrentValue : 1),
                EstimatedRecoveryTime = EstimateRecoveryTime(tracker),
                RiskLevel = CalculateRiskLevel(tracker),
                RecommendedAction = GetRecommendedAction(tracker)
            };
        }

        private static TimeSpan EstimateRecoveryTime(DrawdownTracker tracker)
        {
            // Simple estimation based on historical recovery patterns
            var recoveryDays = (double)tracker.DrawdownAmount / 100.0; // $100 per day recovery estimate
            return TimeSpan.FromDays(Math.Max(1, recoveryDays));
        }

        private static string CalculateRiskLevel(DrawdownTracker tracker)
        {
            return tracker.DrawdownPercent switch
            {
                < 5 => "LOW",
                < 10 => "MODERATE",
                < 20 => "HIGH",
                _ => "CRITICAL"
            };
        }

        private static string GetRecommendedAction(DrawdownTracker tracker)
        {
            return tracker.DrawdownAmount switch
            {
                < 250 => "Monitor closely",
                < 500 => "Reduce position size",
                < 1000 => "Switch to conservative mode",
                < 1500 => "Halt new trades",
                _ => "Emergency stop"
            };
        }
        
        private static void LogAction(string message) => Console.WriteLine($"[DrawdownProtection] {message}");
        private static void LogWarning(string message) => Console.WriteLine($"[DrawdownProtection] WARNING: {message}");
        private static void LogCritical(string message) => Console.WriteLine($"[DrawdownProtection] CRITICAL: {message}");
        private static void LogInfo(string message) => Console.WriteLine($"[DrawdownProtection] INFO: {message}");
    }
}
