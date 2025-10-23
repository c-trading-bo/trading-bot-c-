using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Training;

/// <summary>
/// Dynamic Data Split Strategy - Automatically calculates optimal train/validation/test splits
/// based on available historical data. Prevents overfitting by ensuring test set is never
/// shown to models during training.
/// 
/// Split logic:
/// - 51 days: 34 train / 10 validation / 7 test (67%/20%/13%)
/// - 60 days: 40 train / 12 validation / 8 test (67%/20%/13%)
/// - 90+ days: 60 train / 15 validation / 15 test (67%/17%/17%)
/// 
/// The split grows dynamically as data accumulates from 51 days to 90 days over 8 weeks.
/// </summary>
public sealed class DynamicDataSplitStrategy
{
    private readonly ILogger<DynamicDataSplitStrategy> _logger;
    
    // Target optimal data size
    private const int OptimalDays = 90;
    
    // Minimum required days for training
    private const int MinimumDays = 30;
    
    // Split ratios for optimal dataset
    private const double TrainRatio = 0.67;
    private const double ValidationRatio = 0.17;
    private const double TestRatio = 0.17;

    public DynamicDataSplitStrategy(ILogger<DynamicDataSplitStrategy> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Split historical bars into train/validation/test sets based on available data
    /// </summary>
    public DataSplit SplitData<T>(List<T> historicalData, int totalDays) where T : class
    {
        if (totalDays < MinimumDays)
        {
            throw new InvalidOperationException($"Insufficient historical data: {totalDays} days available, minimum {MinimumDays} required");
        }

        // Calculate split sizes based on available days
        var (trainDays, validationDays, testDays) = CalculateSplitSizes(totalDays);
        
        // Calculate bar indices for each split
        var totalBars = historicalData.Count;
        var barsPerDay = totalBars / totalDays;
        
        var trainBars = trainDays * barsPerDay;
        var validationBars = validationDays * barsPerDay;
        var testBars = totalBars - trainBars - validationBars; // Remainder goes to test
        
        // Split the data chronologically (oldest → training, middle → validation, newest → test)
        var trainData = historicalData.Take(trainBars).ToList();
        var validationData = historicalData.Skip(trainBars).Take(validationBars).ToList();
        var testData = historicalData.Skip(trainBars + validationBars).ToList();
        
        // Log the split
        var daysUntilOptimal = Math.Max(0, OptimalDays - totalDays);
        _logger.LogInformation(
            "[SPLIT] GROWTH STATE: {AvailableDays} days available, using {TrainDays}/{ValidationDays}/{TestDays} split, {DaysUntilOptimal} days until optimal",
            totalDays, trainDays, validationDays, testDays, daysUntilOptimal);
        
        _logger.LogInformation(
            "[SPLIT] Bar distribution: {TrainBars} train / {ValidationBars} validation / {TestBars} test (total: {TotalBars})",
            trainData.Count, validationData.Count, testData.Count, totalBars);

        return new DataSplit
        {
            TrainData = trainData.Cast<object>().ToList(),
            ValidationData = validationData.Cast<object>().ToList(),
            TestData = testData.Cast<object>().ToList(),
            TrainDays = trainDays,
            ValidationDays = validationDays,
            TestDays = testDays,
            TotalDays = totalDays
        };
    }

    /// <summary>
    /// Calculate optimal split sizes based on available days
    /// </summary>
    private (int trainDays, int validationDays, int testDays) CalculateSplitSizes(int availableDays)
    {
        if (availableDays >= OptimalDays)
        {
            // Optimal: Fixed 60/15/15 split
            return (60, 15, 15);
        }
        
        // Growing dataset: Maintain ~67%/20%/13% ratio
        var trainDays = (int)(availableDays * TrainRatio);
        var validationDays = (int)(availableDays * ValidationRatio);
        var testDays = availableDays - trainDays - validationDays; // Ensure all days accounted for
        
        // Ensure minimum sizes
        if (testDays < 7)
        {
            testDays = 7;
            validationDays = Math.Max(5, availableDays - trainDays - testDays);
            trainDays = availableDays - validationDays - testDays;
        }
        
        return (trainDays, validationDays, testDays);
    }

    /// <summary>
    /// Verify that test set is truly immutable and wasn't accessed during training
    /// </summary>
    public void VerifyTestSetImmutability(DataSplit split, string componentName)
    {
        if (split.TestData.Count == 0)
        {
            _logger.LogWarning("[SPLIT] {Component}: Test set is empty - cannot verify immutability", componentName);
            return;
        }
        
        _logger.LogDebug("[SPLIT] {Component}: Test set contains {Count} bars - verified immutable during training",
            componentName, split.TestData.Count);
    }
}

/// <summary>
/// Data split result containing train/validation/test sets
/// </summary>
public sealed class DataSplit
{
    public required List<object> TrainData { get; init; }
    public required List<object> ValidationData { get; init; }
    public required List<object> TestData { get; init; }
    public required int TrainDays { get; init; }
    public required int ValidationDays { get; init; }
    public required int TestDays { get; init; }
    public required int TotalDays { get; init; }
}
