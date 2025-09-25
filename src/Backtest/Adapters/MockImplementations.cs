using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.Backtest;

namespace TradingBot.Backtest.Adapters
{
    /// <summary>
    /// Mock historical data provider for initial integration
    /// In production: Replace with TopstepX API integration or database provider
    /// </summary>
    public class MockHistoricalDataProvider : IHistoricalDataProvider
    {
        private readonly ILogger<MockHistoricalDataProvider> _logger;
        
        public MockHistoricalDataProvider(ILogger<MockHistoricalDataProvider> logger)
        {
            _logger = logger;
        }

        public async Task<IAsyncEnumerable<Quote>> GetHistoricalQuotesAsync(
            string symbol, 
            DateTime startTime, 
            DateTime endTime, 
            CancellationToken cancellationToken = default)
        {
            _logger.LogDebug("Loading historical data for {Symbol} from {StartTime} to {EndTime}", symbol, startTime, endTime);
            await Task.CompletedTask;
            return GenerateMockQuotes(symbol, startTime, endTime);
        }

        public async Task<bool> IsDataAvailableAsync(
            string symbol, 
            DateTime startTime, 
            DateTime endTime, 
            CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask;
            // Mock always returns true - in production would check data availability
            return true;
        }

        public async Task<(DateTime EarliestData, DateTime LatestData)> GetDataRangeAsync(
            string symbol, 
            CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask;
            // Mock returns 2-year range - in production would query actual data range
            return (DateTime.UtcNow.AddYears(-2), DateTime.UtcNow.Date);
        }

        private async IAsyncEnumerable<Quote> GenerateMockQuotes(string symbol, DateTime startTime, DateTime endTime)
        {
            await Task.CompletedTask;
            
            var current = startTime;
            var basePrice = GetBasePriceForSymbol(symbol);
            var random = new Random(startTime.GetHashCode() ^ symbol.GetHashCode());

            _logger.LogDebug("Generating mock quotes for {Symbol} from {StartTime} to {EndTime} (base price: {BasePrice})", 
                symbol, startTime, endTime, basePrice);

            while (current <= endTime)
            {
                // Generate realistic price movement
                var change = (decimal)((random.NextDouble() - 0.5) * 20); // +/- $10
                basePrice += change;
                basePrice = Math.Max(basePrice * 0.95m, Math.Min(basePrice * 1.05m, basePrice)); // Keep within 5% range

                var spread = basePrice * 0.0002m; // 0.02% spread
                var bid = basePrice - spread / 2;
                var ask = basePrice + spread / 2;

                yield return new Quote(
                    Time: current,
                    Symbol: symbol,
                    Bid: Math.Round(bid, 2),
                    Ask: Math.Round(ask, 2),
                    Last: Math.Round(basePrice, 2),
                    Volume: random.Next(50, 500),
                    Open: Math.Round(basePrice, 2),
                    High: Math.Round(basePrice + (decimal)(random.NextDouble() * 5), 2),
                    Low: Math.Round(basePrice - (decimal)(random.NextDouble() * 5), 2),
                    Close: Math.Round(basePrice, 2)
                );

                current = current.AddMinutes(1); // 1-minute bars
                
                if (current.Hour % 4 == 0 && current.Minute == 0) // Log progress every 4 hours
                {
                    _logger.LogDebug("Generated quotes up to {Current} for {Symbol}", current, symbol);
                }
            }
        }

        private static decimal GetBasePriceForSymbol(string symbol)
        {
            return symbol switch
            {
                "ES" => 4500.00m,
                "MES" => 4500.00m,
                "NQ" => 15000.00m,
                "MNQ" => 15000.00m,
                "YM" => 35000.00m,
                "MYM" => 35000.00m,
                _ => 100.00m
            };
        }
    }

    /// <summary>
    /// Mock model registry for initial integration
    /// In production: Replace with actual model storage system
    /// </summary>
    public class MockModelRegistry : IModelRegistry
    {
        private readonly ILogger<MockModelRegistry> _logger;
        private readonly Dictionary<string, List<ModelCard>> _modelsByFamily = new();
        
        public MockModelRegistry(ILogger<MockModelRegistry> logger)
        {
            _logger = logger;
            InitializeDefaultModels();
        }

        public async Task<ModelCard?> GetModelAsOfDateAsync(
            string familyName, 
            DateTime asOfDate, 
            CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask;
            
            if (!_modelsByFamily.TryGetValue(familyName, out var models))
            {
                _logger.LogWarning("No models found for family {FamilyName}", familyName);
                return null;
            }

            // Get the most recent model trained before the asOfDate
            var historicalModel = models
                .Where(m => m.TrainedAt < asOfDate)
                .OrderByDescending(m => m.TrainedAt)
                .FirstOrDefault();

            if (historicalModel != null)
            {
                _logger.LogDebug("Found historical model {ModelId} for {FamilyName} as of {AsOfDate}", 
                    historicalModel.ModelId, familyName, asOfDate);
            }
            else
            {
                _logger.LogWarning("No historical model found for {FamilyName} as of {AsOfDate}", familyName, asOfDate);
            }

            return historicalModel;
        }

        public async Task<ModelPaths?> GetModelPathsAsync(string modelId, CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask;
            
            return new ModelPaths(
                OnnxModelPath: $"/models/{modelId}.onnx",
                ConfigPath: $"/models/{modelId}_config.json",
                MetadataPath: $"/models/{modelId}_metadata.json"
            );
        }

        public async Task<bool> RegisterModelAsync(ModelCard modelCard, CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask;
            
            if (!_modelsByFamily.ContainsKey(modelCard.FamilyName))
            {
                _modelsByFamily[modelCard.FamilyName] = new List<ModelCard>();
            }
            
            _modelsByFamily[modelCard.FamilyName].Add(modelCard);
            _logger.LogInformation("Registered model {ModelId} for family {FamilyName}", modelCard.ModelId, modelCard.FamilyName);
            
            return true;
        }

        public async Task<List<ModelCard>> ListModelsAsync(string familyName, CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask;
            
            return _modelsByFamily.TryGetValue(familyName, out var models) 
                ? models.OrderByDescending(m => m.TrainedAt).ToList()
                : new List<ModelCard>();
        }

        public async Task<bool> ModelExistsAsync(string modelId, CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask;
            
            return _modelsByFamily.Values
                .SelectMany(models => models)
                .Any(m => m.ModelId == modelId);
        }

        private void InitializeDefaultModels()
        {
            // Create some default historical models for testing
            var baseDate = DateTime.UtcNow.AddDays(-100);
            
            var families = new[] { "ESStrategy", "MomentumModel", "TestStrategy" };
            
            foreach (var family in families)
            {
                _modelsByFamily[family] = new List<ModelCard>();
                
                // Create models every 7 days for the past 100 days
                for (int i = 0; i < 14; i++)
                {
                    var trainedAt = baseDate.AddDays(i * 7);
                    var modelId = $"{family}_{trainedAt:yyyyMMdd}_{i:D2}";
                    
                    _modelsByFamily[family].Add(new ModelCard(
                        ModelId: modelId,
                        FamilyName: family,
                        Version: $"1.{i}",
                        TrainedAt: trainedAt,
                        TrainingDataStart: trainedAt.AddDays(-30),
                        TrainingDataEnd: trainedAt,
                        Metrics: new Dictionary<string, double>
                        {
                            ["accuracy"] = 0.6 + (i * 0.01), // Improving accuracy over time
                            ["f1_score"] = 0.55 + (i * 0.01),
                            ["sharpe_ratio"] = 0.3 + (i * 0.02)
                        },
                        ModelPath: $"/models/{modelId}.onnx",
                        ConfigPath: $"/models/{modelId}_config.json",
                        IsActive: i == 13 // Latest model is active
                    ));
                }
            }

            _logger.LogInformation("Initialized mock model registry with {FamilyCount} families and {TotalModels} models", 
                _modelsByFamily.Count, _modelsByFamily.Values.Sum(m => m.Count));
        }
    }
}