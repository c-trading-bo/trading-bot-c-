using System;
using System.IO;
using System.Text.Json;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore.MetaLabeler;

/// <summary>
/// Manages walk-forward validation and model training with embargo periods
/// to prevent lookahead bias in supervised ML models.
/// </summary>
public class WalkForwardTrainer
{
    private readonly IHistoricalDataProvider _dataProvider;
    private readonly TripleBarrierLabeler _labeler;
    private readonly string _modelsPath;
    private readonly WalkForwardConfig _config;

    public WalkForwardTrainer(
        IHistoricalDataProvider dataProvider,
        TripleBarrierLabeler labeler,
        string modelsPath,
        WalkForwardConfig? config = null)
    {
        _dataProvider = dataProvider;
        _labeler = labeler;
        _modelsPath = modelsPath;
        _config = config ?? new WalkForwardConfig();

        Directory.CreateDirectory(_modelsPath);
    }

    /// <summary>
    /// Runs walk-forward validation training process with embargo periods.
    /// </summary>
    public async Task<WalkForwardResults> RunWalkForwardTrainingAsync(
        DateTime startDate,
        DateTime endDate,
        CancellationToken ct = default)
    {
        var results = new WalkForwardResults
        {
            StartDate = startDate,
            EndDate = endDate,
            Config = _config,
            Folds = new List<ValidationFold>()
        };

        var currentDate = startDate;
        var foldNumber = 0;

        Console.WriteLine($"[WALK-FORWARD] Starting training from {startDate:yyyy-MM-dd} to {endDate:yyyy-MM-dd}");
        Console.WriteLine($"[WALK-FORWARD] Training window: {_config.TrainingWindowDays} days, " +
                         $"Test window: {_config.TestWindowDays} days, " +
                         $"Embargo: {_config.EmbargoHours} hours");

        while (currentDate.AddDays(_config.TrainingWindowDays + _config.TestWindowDays) <= endDate)
        {
            foldNumber++;
            Console.WriteLine($"[WALK-FORWARD] Processing fold {foldNumber}...");

            var fold = await ProcessFoldAsync(currentDate, foldNumber, ct);
            results.Folds.Add(fold);

            // Move to next fold
            currentDate = currentDate.AddDays(_config.TestWindowDays);

            if (ct.IsCancellationRequested)
                break;
        }

        // Calculate overall results
        results.OverallMetrics = CalculateOverallMetrics(results.Folds);

        Console.WriteLine($"[WALK-FORWARD] Completed {foldNumber} folds. " +
                         $"Overall accuracy: {results.OverallMetrics.Accuracy:P1}, " +
                         $"Brier score: {results.OverallMetrics.BrierScore:F3}");

        await SaveResultsAsync(results, ct);
        return results;
    }

    private async Task<ValidationFold> ProcessFoldAsync(
        DateTime foldStartDate,
        int foldNumber,
        CancellationToken ct)
    {
        var trainStart = foldStartDate;
        var trainEnd = trainStart.AddDays(_config.TrainingWindowDays);
        var embargoEnd = trainEnd.AddHours(_config.EmbargoHours);
        var testStart = embargoEnd;
        var testEnd = testStart.AddDays(_config.TestWindowDays);

        Console.WriteLine($"[WALK-FORWARD] Fold {foldNumber}: " +
                         $"Train {trainStart:MM-dd} to {trainEnd:MM-dd}, " +
                         $"Test {testStart:MM-dd} to {testEnd:MM-dd}");

        var fold = new ValidationFold
        {
            FoldNumber = foldNumber,
            TrainStart = trainStart,
            TrainEnd = trainEnd,
            EmbargoStart = trainEnd,
            EmbargoEnd = embargoEnd,
            TestStart = testStart,
            TestEnd = testEnd
        };

        try
        {
            // Get training data
            var trainSignals = await GetHistoricalSignalsAsync(trainStart, trainEnd, ct);
            var trainLabeled = await _labeler.LabelSignalsAsync(trainSignals, ct);

            fold.TrainingSamples = trainLabeled.Count;

            if (trainLabeled.Count < _config.MinTrainingSamples)
            {
                fold.Status = FoldStatus.InsufficientData;
                fold.ErrorMessage = $"Only {trainLabeled.Count} training samples, need {_config.MinTrainingSamples}";
                return fold;
            }

            // Train model (placeholder - would integrate with ML framework)
            var modelPath = await TrainModelAsync(trainLabeled, foldNumber, ct);
            fold.ModelPath = modelPath;

            // Get test data (respecting embargo period)
            var testSignals = await GetHistoricalSignalsAsync(testStart, testEnd, ct);
            var testLabeled = await _labeler.LabelSignalsAsync(testSignals, ct);

            fold.TestSamples = testLabeled.Count;

            if (testLabeled.Count == 0)
            {
                fold.Status = FoldStatus.NoTestData;
                return fold;
            }

            // Evaluate model on test set
            var metrics = await EvaluateModelAsync(modelPath, testLabeled, ct);
            fold.Metrics = metrics;
            fold.Status = FoldStatus.Completed;

            Console.WriteLine($"[WALK-FORWARD] Fold {foldNumber} complete: " +
                             $"Train={fold.TrainingSamples}, Test={fold.TestSamples}, " +
                             $"Accuracy={metrics.Accuracy:P1}, Brier={metrics.BrierScore:F3}");
        }
        catch (Exception ex)
        {
            fold.Status = FoldStatus.Error;
            fold.ErrorMessage = ex.Message;
            Console.WriteLine($"[WALK-FORWARD] Fold {foldNumber} failed: {ex.Message}");
        }

        return fold;
    }

    private async Task<List<HistoricalTradeSignal>> GetHistoricalSignalsAsync(
        DateTime start,
        DateTime end,
        CancellationToken ct)
    {
        // This would integrate with your signal history storage
        // For now, return empty list as placeholder
        await Task.CompletedTask;
        return new List<HistoricalTradeSignal>();
    }

    private async Task<string> TrainModelAsync(
        List<LabeledTradeData> trainData,
        int foldNumber,
        CancellationToken ct)
    {
        // Placeholder for actual ML training
        // Would integrate with Python training scripts or C# ML.NET

        var modelPath = Path.Combine(_modelsPath, $"meta_model_fold_{foldNumber}.onnx");

        // Export training data for external training
        var trainingDataPath = Path.Combine(_modelsPath, $"training_data_fold_{foldNumber}.json");
        var trainingJson = JsonSerializer.Serialize(trainData, new JsonSerializerOptions
        {
            WriteIndented = true
        });

        await File.WriteAllTextAsync(trainingDataPath, trainingJson, ct);

        Console.WriteLine($"[WALK-FORWARD] Exported {trainData.Count} training samples to {trainingDataPath}");
        Console.WriteLine($"[WALK-FORWARD] Model training placeholder - would generate {modelPath}");

        // Create placeholder model file
        await File.WriteAllTextAsync(modelPath, "# Placeholder ONNX model", ct);

        return modelPath;
    }

    private async Task<ValidationMetrics> EvaluateModelAsync(
        string modelPath,
        List<LabeledTradeData> testData,
        CancellationToken ct)
    {
        await Task.CompletedTask;

        // Placeholder evaluation - would load ONNX model and predict
        var random = new Random(42);
        var predictions = testData.Select(d => (decimal)random.NextDouble()).ToList();
        var actuals = testData.Select(d => d.Label).ToList();

        return CalculateMetrics(predictions, actuals);
    }

    private ValidationMetrics CalculateMetrics(List<decimal> predictions, List<decimal> actuals)
    {
        if (predictions.Count != actuals.Count || predictions.Count == 0)
        {
            return new ValidationMetrics();
        }

        var correct = 0;
        var brierSum = 0m;
        var logLossSum = 0m;

        for (int i = 0; i < predictions.Count; i++)
        {
            var pred = Math.Max(0.001m, Math.Min(0.999m, predictions[i])); // Clip for stability
            var actual = actuals[i];
            var actualBinary = actual > 0.5m ? 1m : 0m;
            var predBinary = pred > 0.5m ? 1m : 0m;

            if (predBinary == actualBinary)
                correct++;

            // Brier score
            brierSum += (pred - actual) * (pred - actual);

            // Log loss
            if (actualBinary == 1m)
                logLossSum -= (decimal)Math.Log((double)pred);
            else
                logLossSum -= (decimal)Math.Log((double)(1 - pred));
        }

        return new ValidationMetrics
        {
            Accuracy = (decimal)correct / predictions.Count,
            BrierScore = brierSum / predictions.Count,
            LogLoss = logLossSum / predictions.Count,
            SampleCount = predictions.Count,
            PositiveRate = actuals.Count(a => a > 0.5m) / (decimal)actuals.Count
        };
    }

    private ValidationMetrics CalculateOverallMetrics(List<ValidationFold> folds)
    {
        var completedFolds = folds.Where(f => f.Status == FoldStatus.Completed && f.Metrics != null).ToList();

        if (!completedFolds.Any())
        {
            return new ValidationMetrics();
        }

        var totalSamples = completedFolds.Sum(f => f.Metrics!.SampleCount);
        var weightedAccuracy = completedFolds.Sum(f => f.Metrics!.Accuracy * f.Metrics.SampleCount) / totalSamples;
        var weightedBrier = completedFolds.Sum(f => f.Metrics!.BrierScore * f.Metrics.SampleCount) / totalSamples;
        var weightedLogLoss = completedFolds.Sum(f => f.Metrics!.LogLoss * f.Metrics.SampleCount) / totalSamples;

        return new ValidationMetrics
        {
            Accuracy = weightedAccuracy,
            BrierScore = weightedBrier,
            LogLoss = weightedLogLoss,
            SampleCount = totalSamples,
            PositiveRate = completedFolds.Average(f => f.Metrics!.PositiveRate)
        };
    }

    private async Task SaveResultsAsync(WalkForwardResults results, CancellationToken ct)
    {
        var resultsPath = Path.Combine(_modelsPath, $"walk_forward_results_{DateTime.UtcNow:yyyyMMdd_HHmmss}.json");
        var json = JsonSerializer.Serialize(results, new JsonSerializerOptions
        {
            WriteIndented = true,
            Converters = { new DateTimeJsonConverter() }
        });

        await File.WriteAllTextAsync(resultsPath, json, ct);
        Console.WriteLine($"[WALK-FORWARD] Results saved to {resultsPath}");
    }
}

/// <summary>
/// Walk-forward training configuration
/// </summary>
public record WalkForwardConfig
{
    public int TrainingWindowDays { get; init; } = 90; // 3 months training
    public int TestWindowDays { get; init; } = 30;     // 1 month testing
    public int EmbargoHours { get; init; } = 24;       // 1 day embargo
    public int MinTrainingSamples { get; init; } = 100; // Minimum samples for training
    public decimal ValidationSplit { get; init; } = 0.2m; // 20% for validation
}

/// <summary>
/// Results of walk-forward training
/// </summary>
public record WalkForwardResults
{
    public DateTime StartDate { get; init; }
    public DateTime EndDate { get; init; }
    public WalkForwardConfig Config { get; init; } = null!;
    public List<ValidationFold> Folds { get; init; } = new();
    public ValidationMetrics OverallMetrics { get; set; } = null!;
    public DateTime CompletedAt { get; init; } = DateTime.UtcNow;
}

/// <summary>
/// Single validation fold in walk-forward process
/// </summary>
public record ValidationFold
{
    public int FoldNumber { get; init; }
    public DateTime TrainStart { get; init; }
    public DateTime TrainEnd { get; init; }
    public DateTime EmbargoStart { get; init; }
    public DateTime EmbargoEnd { get; init; }
    public DateTime TestStart { get; init; }
    public DateTime TestEnd { get; init; }
    public int TrainingSamples { get; set; }
    public int TestSamples { get; set; }
    public string? ModelPath { get; set; }
    public ValidationMetrics? Metrics { get; set; }
    public FoldStatus Status { get; set; }
    public string? ErrorMessage { get; set; }
}

/// <summary>
/// Validation metrics for model evaluation
/// </summary>
public record ValidationMetrics
{
    public decimal Accuracy { get; init; }
    public decimal BrierScore { get; init; }
    public decimal LogLoss { get; init; }
    public int SampleCount { get; init; }
    public decimal PositiveRate { get; init; }
}

/// <summary>
/// Status of a validation fold
/// </summary>
public enum FoldStatus
{
    Pending,
    InsufficientData,
    NoTestData,
    Completed,
    Error
}

/// <summary>
/// Custom DateTime JSON converter
/// </summary>
public class DateTimeJsonConverter : System.Text.Json.Serialization.JsonConverter<DateTime>
{
    public override DateTime Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        return DateTime.Parse(reader.GetString()!);
    }

    public override void Write(Utf8JsonWriter writer, DateTime value, JsonSerializerOptions options)
    {
        writer.WriteStringValue(value.ToString("yyyy-MM-ddTHH:mm:ss.fffZ"));
    }
}
