using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Linq;
using System.Collections.Generic;

namespace BotCore.MetaLabeler;

/// <summary>
/// ONNX-based meta-labeler for fast win probability estimation in live trading.
/// Uses pre-trained models without Python dependencies.
/// </summary>
public class OnnxMetaLabeler : IMetaLabeler, IDisposable
{
    private readonly InferenceSession _session;
    private readonly CalibrationTracker _calibration;
    private decimal _minWinProbThreshold;

    public OnnxMetaLabeler(string modelPath, decimal minWinProbThreshold = 0.55m)
    {
        _minWinProbThreshold = minWinProbThreshold;
        _calibration = new CalibrationTracker();

        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"ONNX model not found: {modelPath}");
        }

        var sessionOptions = new SessionOptions
        {
            EnableCpuMemArena = false,
            EnableMemoryPattern = false,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };

        _session = new InferenceSession(modelPath, sessionOptions);

        Console.WriteLine($"[META-LABELER] Loaded ONNX model: {Path.GetFileName(modelPath)}");
        Console.WriteLine($"[META-LABELER] Min win probability threshold: {_minWinProbThreshold:P1}");
    }

    public async Task<decimal> EstimateWinProbabilityAsync(
        TradeSignalContext signal,
        MarketContext marketContext,
        CancellationToken ct = default)
    {
        try
        {
            await Task.Delay(1, ct).ConfigureAwait(false); // Satisfy async requirement
            // Prepare features for ONNX model
            var features = PrepareFeatures(signal, marketContext);

            // Create input tensor
            var inputTensor = new DenseTensor<float>(features, new[] { 1, features.Length });
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

            // Run inference
            using var results = _session.Run(inputs);
            var output = results.First().AsEnumerable<float>().First();

            var winProb = Math.Max(0m, Math.Min(1m, (decimal)output));

            Console.WriteLine($"[META-LABELER] {signal.SignalId} p(win)={winProb:P1} " +
                            $"(threshold={_minWinProbThreshold:P1}) " +
                            $"R={signal.RMultiple:F1} confidence={signal.Confidence:F2}");

            return winProb;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[META-LABELER] Error estimating win probability: {ex.Message}");
            return 0.5m; // Neutral probability on error
        }
    }

    public decimal GetMinWinProbabilityThreshold()
    {
        return _minWinProbThreshold;
    }

    public async Task UpdateCalibrationAsync(
        decimal predictedProb,
        bool actualOutcome,
        CancellationToken ct = default)
    {
        _calibration.AddPrediction(predictedProb, actualOutcome);

        // Update threshold based on calibration if needed
        var metrics = await GetCalibrationMetricsAsync(ct).ConfigureAwait(false);
        if (metrics.TotalPredictions > 100 && !metrics.IsWellCalibrated)
        {
            // Adjust threshold if model is poorly calibrated
            var adjustment = metrics.BrierScore > 0.25m ? 0.02m : -0.01m;
            _minWinProbThreshold = Math.Max(0.5m, Math.Min(0.8m, _minWinProbThreshold + adjustment));

            Console.WriteLine($"[META-LABELER] Adjusted threshold to {_minWinProbThreshold:P1} " +
                            $"(Brier={metrics.BrierScore:F3})");
        }
    }

    public async Task<CalibrationMetrics> GetCalibrationMetricsAsync(CancellationToken ct = default)
    {
        return await Task.FromResult(_calibration.GetMetrics()).ConfigureAwait(false);
    }

    private static float[] PrepareFeatures(TradeSignalContext signal, MarketContext marketContext)
    {
        // Feature vector matching training data format
        return new float[]
        {
            (float)signal.AtrMultiple,
            (float)signal.RMultiple,
            (float)signal.Confidence,
            (float)marketContext.SpreadBps,
            (float)marketContext.Volatility,
            (float)marketContext.AtrZScore,
            (float)marketContext.TimeOfDay,
            (float)marketContext.DayOfWeek,
            (float)marketContext.RegimeProbability,
            (float)marketContext.RecentMomentum,
            marketContext.IsSessionStart ? 1f : 0f,
            marketContext.IsSessionEnd ? 1f : 0f,
            signal.IsLong ? 1f : 0f,
            // Add more features as needed
            (float)(marketContext.Volume / 1000), // Normalized volume
            (float)Math.Log((double)marketContext.BidAskSpread + 1) // Log spread
        };
    }

    public void Dispose()
    {
        _session?.Dispose();
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// Tracks model calibration metrics in real-time
/// </summary>
internal class CalibrationTracker
{
    private readonly List<(decimal predicted, bool actual)> _predictions = new();
    private readonly object _lock = new();

    public void AddPrediction(decimal predicted, bool actual)
    {
        lock (_lock)
        {
            _predictions.Add((predicted, actual));

            // Keep only recent predictions (sliding window)
            if (_predictions.Count > 1000)
            {
                _predictions.RemoveAt(0);
            }
        }
    }

    public CalibrationMetrics GetMetrics()
    {
        lock (_lock)
        {
            if (_predictions.Count == 0)
            {
                return new CalibrationMetrics(
                    BrierScore: 0m,
                    LogLoss: 0m,
                    ReliabilityScore: 0m,
                    ResolutionScore: 0m,
                    TotalPredictions: 0,
                    LastUpdated: DateTime.UtcNow,
                    IsWellCalibrated: true
                );
            }

            var brierScore = CalculateBrierScore();
            var logLoss = CalculateLogLoss();
            var reliability = CalculateReliability();
            var resolution = CalculateResolution();

            return new CalibrationMetrics(
                BrierScore: brierScore,
                LogLoss: logLoss,
                ReliabilityScore: reliability,
                ResolutionScore: resolution,
                TotalPredictions: _predictions.Count,
                LastUpdated: DateTime.UtcNow,
                IsWellCalibrated: brierScore < 0.2m && reliability < 0.05m
            );
        }
    }

    private decimal CalculateBrierScore()
    {
        var sum = _predictions.Sum(p =>
        {
            var actual = p.actual ? 1m : 0m;
            return (p.predicted - actual) * (p.predicted - actual);
        });
        return sum / _predictions.Count;
    }

    private decimal CalculateLogLoss()
    {
        var sum = _predictions.Sum(p =>
        {
            var prob = Math.Max(0.0001m, Math.Min(0.9999m, p.predicted)); // Clip for numerical stability
            return p.actual
                ? -(decimal)Math.Log((double)prob)
                : -(decimal)Math.Log((double)(1 - prob));
        });
        return sum / _predictions.Count;
    }

    private decimal CalculateReliability()
    {
        // Group predictions into bins and calculate reliability component of Brier score
        const int numBins = 10;
        var bins = new List<decimal>[numBins];
        var outcomes = new List<bool>[numBins];

        for (int i; i < numBins; i++)
        {
            bins[i] = new List<decimal>();
            outcomes[i] = new List<bool>();
        }

        foreach (var (predicted, actual) in _predictions)
        {
            var binIndex = Math.Min(numBins - 1, (int)(predicted * numBins));
            bins[binIndex].Add(predicted);
            outcomes[binIndex].Add(actual);
        }

        decimal reliabilitySum;
        int totalCount = _predictions.Count;

        for (int i; i < numBins; i++)
        {
            if (bins[i].Count == 0) continue;

            var avgPredicted = bins[i].Average();
            var avgActual = outcomes[i].Count > 0 ? outcomes[i].Average(x => x ? 1m : 0m) : 0m;
            var binWeight = (decimal)bins[i].Count / totalCount;

            reliabilitySum += binWeight * (avgPredicted - avgActual) * (avgPredicted - avgActual);
        }

        return reliabilitySum;
    }

    private decimal CalculateResolution()
    {
        var overallMean = _predictions.Average(p => p.actual ? 1m : 0m);

        // Group predictions into bins
        const int numBins = 10;
        var bins = new List<bool>[numBins];

        for (int i; i < numBins; i++)
        {
            bins[i] = new List<bool>();
        }

        foreach (var (predicted, actual) in _predictions)
        {
            var binIndex = Math.Min(numBins - 1, (int)(predicted * numBins));
            bins[binIndex].Add(actual);
        }

        decimal resolutionSum;
        int totalCount = _predictions.Count;

        for (int i; i < numBins; i++)
        {
            if (bins[i].Count == 0) continue;

            var avgActual = bins[i].Average(x => x ? 1m : 0m);
            var binWeight = (decimal)bins[i].Count / totalCount;

            resolutionSum += binWeight * (avgActual - overallMean) * (avgActual - overallMean);
        }

        return resolutionSum;
    }
}
