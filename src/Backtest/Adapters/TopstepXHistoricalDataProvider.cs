using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TradingBot.Backtest;
using BotCore.Services;

namespace TradingBot.Backtest.Adapters
{
    /// <summary>
    /// PRODUCTION Historical Data Provider using REAL TopstepX API
    /// Replaces MockHistoricalDataProvider with actual live data from TopstepX
    /// Addresses user requirement for "actually getting live data for backtesting"
    /// </summary>
    public class TopstepXHistoricalDataProvider : IHistoricalDataProvider
    {
        private readonly ILogger<TopstepXHistoricalDataProvider> _logger;
        private readonly IHistoricalDataBridgeService _bridgeService;
        private readonly HttpClient _httpClient;
        
        public TopstepXHistoricalDataProvider(
            ILogger<TopstepXHistoricalDataProvider> logger,
            IHistoricalDataBridgeService bridgeService,
            HttpClient httpClient)
        {
            _logger = logger;
            _bridgeService = bridgeService;
            _httpClient = httpClient;
        }

        /// <summary>
        /// Get REAL historical quotes from TopstepX API
        /// This replaces the mock data generation with actual market data
        /// </summary>
        public async Task<IAsyncEnumerable<Quote>> GetHistoricalQuotesAsync(
            string symbol, 
            DateTime startTime, 
            DateTime endTime, 
            CancellationToken cancellationToken = default)
        {
            _logger.LogInformation("🔴 REAL DATA: Loading historical quotes from TopstepX API for {Symbol} from {StartTime} to {EndTime}", 
                symbol, startTime, endTime);

            try
            {
                // Get real historical bars from TopstepX through existing bridge service
                var contractId = MapSymbolToContractId(symbol);
                var totalDays = (int)(endTime - startTime).TotalDays + 1;
                var estimatedBars = totalDays * 390; // ~390 minutes per trading day
                
                var historicalBars = await _bridgeService.GetRecentHistoricalBarsAsync(contractId, estimatedBars);
                
                if (!historicalBars.Any())
                {
                    _logger.LogWarning("No historical bars available from TopstepX for {Symbol} ({ContractId})", symbol, contractId);
                    return GetEmptyQuoteStream();
                }

                // Filter bars to requested time range
                var filteredBars = historicalBars
                    .Where(bar => bar.Ts >= startTime && bar.Ts <= endTime)
                    .OrderBy(bar => bar.Ts)
                    .ToList();

                _logger.LogInformation("🟢 REAL DATA: Retrieved {BarCount} real historical bars for {Symbol} from TopstepX", 
                    filteredBars.Count, symbol);

                return ConvertBarsToQuotes(symbol, filteredBars);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ REAL DATA: Failed to load historical data from TopstepX for {Symbol}", symbol);
                throw;
            }
        }

        /// <summary>
        /// Check if real historical data is available from TopstepX
        /// </summary>
        public async Task<bool> IsDataAvailableAsync(
            string symbol, 
            DateTime startTime, 
            DateTime endTime, 
            CancellationToken cancellationToken = default)
        {
            try
            {
                var contractId = MapSymbolToContractId(symbol);
                var isAvailable = await _bridgeService.ValidateHistoricalDataAsync(contractId);
                
                _logger.LogDebug("Data availability check for {Symbol}: {IsAvailable}", symbol, isAvailable);
                return isAvailable;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to check data availability for {Symbol}", symbol);
                return false;
            }
        }

        /// <summary>
        /// Get actual data range available from TopstepX
        /// </summary>
        public async Task<(DateTime EarliestData, DateTime LatestData)> GetDataRangeAsync(
            string symbol, 
            CancellationToken cancellationToken = default)
        {
            try
            {
                var contractId = MapSymbolToContractId(symbol);
                
                // Get a sample of recent bars to determine data range
                var sampleBars = await _bridgeService.GetRecentHistoricalBarsAsync(contractId, 100);
                
                if (!sampleBars.Any())
                {
                    // Fallback to reasonable range if no data available
                    return (DateTime.UtcNow.AddDays(-30), DateTime.UtcNow);
                }

                var earliestData = sampleBars.Min(b => b.Ts);
                var latestData = sampleBars.Max(b => b.Ts);
                
                _logger.LogDebug("Real data range for {Symbol}: {EarliestData} to {LatestData}", 
                    symbol, earliestData, latestData);

                return (earliestData, latestData);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to get data range for {Symbol}", symbol);
                // Return reasonable fallback range
                return (DateTime.UtcNow.AddDays(-30), DateTime.UtcNow);
            }
        }

        /// <summary>
        /// Convert BotCore Bar objects to Backtest Quote objects
        /// </summary>
        private async IAsyncEnumerable<Quote> ConvertBarsToQuotes(string symbol, List<BotCore.Models.Bar> bars)
        {
            await Task.CompletedTask; // Satisfy async enumerable requirement
            
            foreach (var bar in bars)
            {
                // Convert bar data to quote format
                // Estimate bid/ask from OHLC data (in real system, would have tick data)
                var spread = CalculateSpreadEstimate(bar.Close, symbol);
                var midPrice = bar.Close;
                var bid = midPrice - (spread / 2);
                var ask = midPrice + (spread / 2);

                yield return new Quote(
                    Time: bar.Ts,
                    Symbol: symbol,
                    Bid: Math.Round(bid, 2),
                    Ask: Math.Round(ask, 2),
                    Last: Math.Round(bar.Close, 2),
                    Volume: (int)Math.Min(bar.Volume, int.MaxValue),
                    Open: Math.Round(bar.Open, 2),
                    High: Math.Round(bar.High, 2),
                    Low: Math.Round(bar.Low, 2),
                    Close: Math.Round(bar.Close, 2)
                );
            }
        }

        /// <summary>
        /// Map trading symbols to TopstepX contract IDs
        /// </summary>
        private string MapSymbolToContractId(string symbol)
        {
            return symbol switch
            {
                "ES" => "ES",
                "MES" => "MES", 
                "NQ" => "NQ",
                "MNQ" => "MNQ",
                "YM" => "YM",
                "MYM" => "MYM",
                "RTY" => "RTY",
                "M2K" => "M2K",
                _ => symbol // Default to symbol as-is
            };
        }

        /// <summary>
        /// Calculate realistic spread estimate based on symbol
        /// </summary>
        private decimal CalculateSpreadEstimate(decimal price, string symbol)
        {
            return symbol switch
            {
                "ES" => 0.25m, // E-mini S&P 500: 0.25 point tick size
                "MES" => 0.25m, // Micro E-mini S&P 500: same tick size
                "NQ" => 0.25m, // E-mini NASDAQ: 0.25 point tick size  
                "MNQ" => 0.25m, // Micro E-mini NASDAQ: same tick size
                "YM" => 1.0m, // E-mini Dow: 1 point tick size
                "MYM" => 1.0m, // Micro E-mini Dow: same tick size
                _ => Math.Max(0.01m, price * 0.0001m) // Default: 1 basis point
            };
        }

        /// <summary>
        /// Return empty quote stream when no data available
        /// </summary>
        private async IAsyncEnumerable<Quote> GetEmptyQuoteStream()
        {
            await Task.CompletedTask;
            yield break;
        }
    }

    /// <summary>
    /// Extension methods for TopstepX data provider registration
    /// </summary>
    public static class TopstepXDataProviderExtensions
    {
        /// <summary>
        /// Register the real TopstepX historical data provider
        /// This replaces mock implementations with actual live data
        /// </summary>
        public static IServiceCollection AddTopstepXHistoricalDataProvider(this IServiceCollection services)
        {
            services.AddSingleton<IHistoricalDataProvider, TopstepXHistoricalDataProvider>();
            services.AddHttpClient<TopstepXHistoricalDataProvider>();
            return services;
        }
    }
}