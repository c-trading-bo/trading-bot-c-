using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TradingBot.Backtest.UI
{
    /// <summary>
    /// Console UI for backtest tick replay visualization with live position tracking
    /// 
    /// USAGE:
    /// Set environment variable: export ENABLE_BACKTEST_UI=1
    /// Or in appsettings.backtest.json: "EnableTickReplayUI": true
    /// 
    /// FEATURES:
    /// - Depth of Market (DOM) - Last 10 ticks with bid/ask/volume
    /// - Bot Brain Thinking - Signal detection with entry/stop/target details
    /// - Open Position Panel - Live P&L tracking with visual indicators
    /// - Account Stats - Equity, daily P&L, trade statistics
    /// 
    /// CONFIGURATION:
    /// - ReplaySpeed: 1 = real-time, 2 = 2x speed, etc.
    /// - Automatically shows/hides position panel based on trade status
    /// - Renders every 5 ticks or when signals detected
    /// 
    /// The UI plays through all historical bars in real-time, allowing you to see
    /// how well the bot performs on historical data and validates training improvements.
    /// </summary>
    public class BacktestConsoleUI
    {
        private readonly string _symbol;
        private readonly DateTime _backtestDate;
        private readonly Queue<TickDisplay> _recentTicks = new(10);
        private decimal _currentEquity;
        private decimal _dailyPnL;
        private decimal _openPnL;
        private int _totalTrades;
        private int _winningTrades;
        private decimal _bestTrade;
        private string _botThinking = "Initializing...";
        private decimal _currentBid;
        private decimal _currentAsk;
        private int _replaySpeed = 1;
        
        // Open position tracking
        private PositionInfo? _openPosition;
        private decimal _currentPrice;
        
        public BacktestConsoleUI(string symbol, DateTime backtestDate, decimal initialEquity)
        {
            _symbol = symbol;
            _backtestDate = backtestDate;
            _currentEquity = initialEquity;
            _dailyPnL = 0m;
            _openPnL = 0m;
        }

        /// <summary>
        /// Clear the console and render the full UI
        /// </summary>
        public void Render()
        {
            Console.Clear();
            Console.OutputEncoding = Encoding.UTF8;
            
            RenderHeader();
            RenderDepthOfMarket();
            RenderBotThinking();
            if (_openPosition != null)
            {
                RenderOpenPosition();
            }
            RenderAccountStats();
        }

        /// <summary>
        /// Add a new tick to the display
        /// </summary>
        public void AddTick(DateTime timestamp, decimal price, int volume, string direction, decimal bid, decimal ask)
        {
            _currentBid = bid;
            _currentAsk = ask;
            _currentPrice = price;
            
            _recentTicks.Enqueue(new TickDisplay
            {
                Timestamp = timestamp,
                Price = price,
                Volume = volume,
                Direction = direction,
                Bid = bid,
                Ask = ask
            });
            
            while (_recentTicks.Count > 10)
                _recentTicks.Dequeue();
        }

        /// <summary>
        /// Update bot thinking status
        /// </summary>
        public void UpdateBotThinking(string thinking)
        {
            _botThinking = thinking;
        }

        /// <summary>
        /// Update account statistics
        /// </summary>
        public void UpdateAccountStats(decimal equity, decimal dailyPnL, decimal openPnL, int totalTrades, int winningTrades, decimal bestTrade)
        {
            _currentEquity = equity;
            _dailyPnL = dailyPnL;
            _openPnL = openPnL;
            _totalTrades = totalTrades;
            _winningTrades = winningTrades;
            _bestTrade = bestTrade;
        }

        /// <summary>
        /// Set replay speed
        /// </summary>
        public void SetReplaySpeed(int speed)
        {
            _replaySpeed = speed;
        }

        /// <summary>
        /// Open a new position
        /// </summary>
        public void OpenPosition(string side, decimal entryPrice, decimal stopLoss, decimal target, int quantity, decimal confidence, string reason)
        {
            _openPosition = new PositionInfo
            {
                Symbol = _symbol,
                Side = side,
                EntryPrice = entryPrice,
                StopLoss = stopLoss,
                Target = target,
                Quantity = quantity,
                Confidence = confidence,
                Reason = reason,
                EntryTime = DateTime.Now
            };
        }

        /// <summary>
        /// Close the current position
        /// </summary>
        public void ClosePosition()
        {
            _openPosition = null;
        }

        /// <summary>
        /// Check if there's an open position
        /// </summary>
        public bool HasOpenPosition()
        {
            return _openPosition != null;
        }

        private void RenderHeader()
        {
            Console.WriteLine("╔══════════════════════════════════════════════════════════════════════╗");
            Console.WriteLine($"║              {_symbol} BACKTEST - LIVE TICK REPLAY{new string(' ', 27 - _symbol.Length)}║");
            Console.WriteLine($"║              {_backtestDate:MMM dd, yyyy  HH:mm:ss} CT{new string(' ', 38)}║");
            Console.WriteLine("╚══════════════════════════════════════════════════════════════════════╝");
            Console.WriteLine();
        }

        private void RenderDepthOfMarket()
        {
            Console.WriteLine("┌─────────────────────────────────────────────────────────────────────┐");
            Console.WriteLine($"│ DEPTH OF MARKET (Last 10 Ticks)                    Speed: {_replaySpeed}x Real   │");
            Console.WriteLine("├─────────────────────────────────────────────────────────────────────┤");
            
            if (_recentTicks.Count == 0)
            {
                Console.WriteLine("│ Waiting for tick data...                                           │");
            }
            else
            {
                foreach (var tick in _recentTicks.Reverse())
                {
                    var dirSymbol = tick.Direction switch
                    {
                        "up" => "↑",
                        "down" => "↓",
                        _ => "→"
                    };
                    
                    var line = $"│ {tick.Timestamp:HH:mm:ss.fff}   {tick.Price,10:N2}   {dirSymbol}   VOL: {tick.Volume,-6}  BID: {tick.Bid,9:N2}  ASK: {tick.Ask,9:N2}";
                    
                    // Pad to 73 characters total (including border)
                    var padding = 73 - line.Length;
                    if (padding > 0)
                        line += new string(' ', padding);
                    line += "│";
                    
                    Console.WriteLine(line);
                }
            }
            
            Console.WriteLine("└─────────────────────────────────────────────────────────────────────┘");
            Console.WriteLine();
        }

        private void RenderBotThinking()
        {
            Console.WriteLine("┌─────────────────────────────────────────────────────────────────────┐");
            Console.WriteLine("│ 🤖 BOT BRAIN THINKING...                                            │");
            Console.WriteLine("├─────────────────────────────────────────────────────────────────────┤");
            
            // Split bot thinking into lines (max 65 chars per line)
            var thinkingLines = WrapText(_botThinking, 65);
            foreach (var line in thinkingLines)
            {
                var paddedLine = $"│ {line}";
                var padding = 73 - paddedLine.Length;
                if (padding > 0)
                    paddedLine += new string(' ', padding);
                paddedLine += "│";
                Console.WriteLine(paddedLine);
            }
            
            // Fill empty space if needed
            while (thinkingLines.Count < 7)
            {
                Console.WriteLine("│                                                                      │");
                thinkingLines.Add("");
            }
            
            Console.WriteLine("└─────────────────────────────────────────────────────────────────────┘");
            Console.WriteLine();
        }

        private void RenderOpenPosition()
        {
            if (_openPosition == null)
                return;

            Console.WriteLine("┌─────────────────────────────────────────────────────────────────────┐");
            Console.WriteLine("│ OPEN POSITION                                                       │");
            Console.WriteLine("├─────────────────────────────────────────────────────────────────────┤");

            // Position details
            var positionLine = $"│ {_openPosition.Symbol} {_openPosition.Side} {_openPosition.Quantity} @ {_openPosition.EntryPrice:N2}";
            var padding = 73 - positionLine.Length;
            if (padding > 0)
                positionLine += new string(' ', padding);
            positionLine += "│";
            Console.WriteLine(positionLine);

            // Current price and direction
            var direction = _currentPrice > _openPosition.EntryPrice ? "↑" : _currentPrice < _openPosition.EntryPrice ? "↓" : "→";
            var priceLine = $"│ Current Price: {_currentPrice:N2}  {direction}";
            padding = 73 - priceLine.Length;
            if (padding > 0)
                priceLine += new string(' ', padding);
            priceLine += "│";
            Console.WriteLine(priceLine);

            // Calculate P&L
            var pnl = _openPosition.Side.ToUpper() == "LONG" 
                ? (_currentPrice - _openPosition.EntryPrice) * _openPosition.Quantity * 50 // ES point value
                : (_openPosition.EntryPrice - _currentPrice) * _openPosition.Quantity * 50;
            var pnlPercent = _openPosition.EntryPrice != 0 ? (pnl / (_openPosition.EntryPrice * _openPosition.Quantity * 50)) * 100 : 0;
            var pnlIndicator = pnl >= 0 ? "🟢" : "🔴";
            var pnlLine = $"│ P&L: {(pnl >= 0 ? "+" : "")}{pnl:C2} ({(pnlPercent >= 0 ? "+" : "")}{pnlPercent:N2}%)  {pnlIndicator}";
            padding = 73 - pnlLine.Length;
            if (padding > 0)
                pnlLine += new string(' ', padding);
            pnlLine += "│";
            Console.WriteLine(pnlLine);

            // Stop loss distance
            var stopDistance = Math.Abs(_openPosition.StopLoss - _openPosition.EntryPrice);
            var stopLine = $"│ Stop: {_openPosition.StopLoss:N2} (-{stopDistance:N1} pts away)";
            padding = 73 - stopLine.Length;
            if (padding > 0)
                stopLine += new string(' ', padding);
            stopLine += "│";
            Console.WriteLine(stopLine);

            // Target distance
            var targetDistance = Math.Abs(_openPosition.Target - _openPosition.EntryPrice);
            var targetLine = $"│ Target: {_openPosition.Target:N2} (+{targetDistance:N1} pts away)";
            padding = 73 - targetLine.Length;
            if (padding > 0)
                targetLine += new string(' ', padding);
            targetLine += "│";
            Console.WriteLine(targetLine);

            Console.WriteLine("└─────────────────────────────────────────────────────────────────────┘");
            Console.WriteLine();
        }

        private void RenderAccountStats()
        {
            var winRate = _totalTrades > 0 ? (_winningTrades * 100.0m / _totalTrades) : 0m;
            var equityChange = _currentEquity - (_currentEquity - _dailyPnL);
            var equityChangePercent = equityChange != 0 && _currentEquity != 0 ? (equityChange / (_currentEquity - equityChange)) * 100 : 0;
            
            Console.WriteLine("┌─────────────────────────────────┬───────────────────────────────────┐");
            Console.WriteLine("│ 💼 ACCOUNT                      │ 📊 TODAY'S STATS                  │");
            Console.WriteLine("├─────────────────────────────────┼───────────────────────────────────┤");
            
            var equityLine = $"│ Equity: ${_currentEquity,9:N0} ({equityChangePercent,+5:N2}%)";
            var tradesLine = $"│ Trades: {_totalTrades,-25}";
            Console.WriteLine(PadLine(equityLine, 35) + PadLine(tradesLine, 35) + "│");
            
            var openPnLLine = $"│ Open P&L: ${_openPnL,13:N2}";
            var winnersLine = $"│ Winners: {_winningTrades} ({winRate,4:N1}%)";
            Console.WriteLine(PadLine(openPnLLine, 35) + PadLine(winnersLine, 35) + "│");
            
            var dailyPnLLine = $"│ Daily P&L: {(_dailyPnL >= 0 ? "+" : "")}{_dailyPnL,10:N2}";
            var pnlLine = $"│ P&L: {(_dailyPnL >= 0 ? "+" : "")}{_dailyPnL,9:N0}";
            Console.WriteLine(PadLine(dailyPnLLine, 35) + PadLine(pnlLine, 35) + "│");
            
            var buyingPowerLine = $"│ Buying Power: ${_currentEquity,10:N0}";
            var bestLine = $"│ Best: {(_bestTrade >= 0 ? "+" : "")}{_bestTrade,7:N0}";
            Console.WriteLine(PadLine(buyingPowerLine, 35) + PadLine(bestLine, 35) + "│");
            
            Console.WriteLine("└─────────────────────────────────┴───────────────────────────────────┘");
            Console.WriteLine();
            
            // Current tick indicator
            if (_recentTicks.Count > 0)
            {
                var lastTick = _recentTicks.Last();
                var dirSymbol = lastTick.Direction switch
                {
                    "up" => "↑",
                    "down" => "↓",
                    _ => "→"
                };
                Console.WriteLine($"🎬 [TICK] Price: {lastTick.Price:N2} {dirSymbol} | Bot sees this tick and analyzes...");
            }
        }

        private static List<string> WrapText(string text, int maxWidth)
        {
            var result = new List<string>();
            if (string.IsNullOrEmpty(text))
            {
                result.Add("");
                return result;
            }
            
            var words = text.Split(' ');
            var currentLine = new StringBuilder();
            
            foreach (var word in words)
            {
                if (currentLine.Length + word.Length + 1 > maxWidth)
                {
                    if (currentLine.Length > 0)
                    {
                        result.Add(currentLine.ToString());
                        currentLine.Clear();
                    }
                    currentLine.Append(word);
                }
                else
                {
                    if (currentLine.Length > 0)
                        currentLine.Append(' ');
                    currentLine.Append(word);
                }
            }
            
            if (currentLine.Length > 0)
                result.Add(currentLine.ToString());
            
            return result;
        }

        private static string PadLine(string line, int width)
        {
            var padding = width - line.Length + 1;
            if (padding > 0)
                return line + new string(' ', padding);
            return line + " ";
        }

        private class TickDisplay
        {
            public DateTime Timestamp { get; set; }
            public decimal Price { get; set; }
            public int Volume { get; set; }
            public string Direction { get; set; } = "";
            public decimal Bid { get; set; }
            public decimal Ask { get; set; }
        }

        private class PositionInfo
        {
            public string Symbol { get; set; } = "";
            public string Side { get; set; } = "";
            public decimal EntryPrice { get; set; }
            public decimal StopLoss { get; set; }
            public decimal Target { get; set; }
            public int Quantity { get; set; }
            public decimal Confidence { get; set; }
            public string Reason { get; set; } = "";
            public DateTime EntryTime { get; set; }
        }
    }
}
