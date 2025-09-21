using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Text.Json;
using System.IO;
using System.Linq;

namespace TradingBot.IntelligenceStack;

/// <summary>
/// Calibration manager with nightly hot-swap capability
/// Fits Platt and isotonic calibration, chooses best by Brier/LogLoss
/// </summary>
public class CalibrationManager : ICalibrationManager, IDisposable
{
    private const int MinCalibrationPoints = 10;
    private const int MinStableCalibrationPoints = 50;
    private const int DefaultThresholdCount = 10;
    private const int ExponentValue = 2;
    
    // LoggerMessage delegates for CA1848 compliance
    private static readonly Action<ILogger, string, Exception?> NoCalibrationMapFound =
        LoggerMessage.Define<string>(LogLevel.Warning, new EventId(3001, "NoCalibrationMapFound"),
            "[CALIBRATION] No calibration map found for model: {ModelId}");
            
    private static readonly Action<ILogger, string, string, Exception?> CalibrationMapLoaded =
        LoggerMessage.Define<string, string>(LogLevel.Debug, new EventId(3002, "CalibrationMapLoaded"),
            "[CALIBRATION] Loaded calibration map for {ModelId} (method: {Method})");
            
    private static readonly Action<ILogger, string, Exception?> CalibrationLoadFailed =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(3003, "CalibrationLoadFailed"),
            "[CALIBRATION] Failed to load calibration map for {ModelId}");
            
    private static readonly Action<ILogger, string, Exception?> CalibrationFitFailed =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(3006, "CalibrationFitFailed"),
            "[CALIBRATION] Failed to fit calibration for {ModelId}");
            
    private static readonly Action<ILogger, string, int, Exception?> InsufficientDataForCalibration =
        LoggerMessage.Define<string, int>(LogLevel.Warning, new EventId(3008, "InsufficientDataForCalibration"),
            "[CALIBRATION] Insufficient data for calibration: {ModelId} ({Count} points)");
            
    private static readonly Action<ILogger, string, string, double, Exception?> CalibrationFittedWithBrier =
        LoggerMessage.Define<string, string, double>(LogLevel.Information, new EventId(3009, "CalibrationFittedWithBrier"),
            "[CALIBRATION] Fitted calibration for {ModelId}: {Method} (Brier: {Brier:F4})");
            
    private static readonly Action<ILogger, string, Exception?> ConfidenceCalibrationFailed =
        LoggerMessage.Define<string>(LogLevel.Error, new EventId(3010, "ConfidenceCalibrationFailed"),
            "[CALIBRATION] Failed to calibrate confidence for {ModelId}");
            
    private static readonly Action<ILogger, DateTime, Exception?> NightlyCalibrationStarted =
        LoggerMessage.Define<DateTime>(LogLevel.Information, new EventId(3011, "NightlyCalibrationStarted"),
            "[CALIBRATION] Starting nightly calibration update at {Time}");
            
    private static readonly Action<ILogger, string, Exception?> ModelsDirectoryNotFound =
        LoggerMessage.Define<string>(LogLevel.Warning, new EventId(3012, "ModelsDirectoryNotFound"),
            "[CALIBRATION] Models directory not found: {Dir}");
            
    private static readonly Action<ILogger, string, Exception?> ModelCalibrationUpdateFailed =
        LoggerMessage.Define<string>(LogLevel.Warning, new EventId(3013, "ModelCalibrationUpdateFailed"),
            "[CALIBRATION] Failed to update calibration for model: {Model}");
            
    private static readonly Action<ILogger, int, Exception?> NightlyCalibrationCompleted =
        LoggerMessage.Define<int>(LogLevel.Information, new EventId(3014, "NightlyCalibrationCompleted"),
            "[CALIBRATION] Nightly calibration completed: {Updated} models updated");
            
    private static readonly Action<ILogger, Exception?> NightlyCalibrationFailed =
        LoggerMessage.Define(LogLevel.Error, new EventId(3015, "NightlyCalibrationFailed"),
            "[CALIBRATION] Nightly calibration failed");
            
    private static readonly Action<ILogger, string, Exception?> CalibrationPointsLoadFailed =
        LoggerMessage.Define<string>(LogLevel.Warning, new EventId(3016, "CalibrationPointsLoadFailed"),
            "[CALIBRATION] Failed to load calibration points for {ModelId}");
            
    private static readonly Action<ILogger, DateTime, Exception?> NightlyCalibrationScheduled =
        LoggerMessage.Define<DateTime>(LogLevel.Information, new EventId(3017, "NightlyCalibrationScheduled"),
            "[CALIBRATION] Scheduled nightly calibration for {Time}");
    
    
    
    private readonly ILogger<CalibrationManager> _logger;
    private readonly string _basePath;
    private readonly Dictionary<string, CalibrationMap> _calibrationCache = new();
    private readonly object _lock = new();
    private Timer? _nightlyTimer;

    public CalibrationManager(ILogger<CalibrationManager> logger, string basePath = "data/calibration")
    {
        _logger = logger;
        _basePath = basePath;
        Directory.CreateDirectory(_basePath);
        
        // Schedule nightly calibration at 02:30
        ScheduleNightlyCalibration();
    }

    public async Task<CalibrationMap> LoadCalibrationMapAsync(string modelId, CancellationToken cancellationToken = default)
    {
        try
        {
            lock (_lock)
            {
                if (_calibrationCache.TryGetValue(modelId, out var cachedMap))
                {
                    return cachedMap;
                }
            }

            var calibrationPath = Path.Combine(_basePath, $"{modelId}_calibration.json");
            if (!File.Exists(calibrationPath))
            {
                NoCalibrationMapFound(_logger, modelId, null);
                return CreateDefaultCalibrationMap(modelId);
            }

            var content = await File.ReadAllTextAsync(calibrationPath, cancellationToken).ConfigureAwait(false);
            var map = JsonSerializer.Deserialize<CalibrationMap>(content);
            
            if (map != null)
            {
                lock (_lock)
                {
                    _calibrationCache[modelId] = map;
                }
                
                CalibrationMapLoaded(_logger, modelId, map.Method.ToString(), null);
                return map;
            }

            return CreateDefaultCalibrationMap(modelId);
        }
        catch (JsonException ex)
        {
            CalibrationLoadFailed(_logger, modelId, ex);
            return CreateDefaultCalibrationMap(modelId);
        }
        catch (IOException ex)
        {
            CalibrationLoadFailed(_logger, modelId, ex);
            return CreateDefaultCalibrationMap(modelId);
        }
        catch (UnauthorizedAccessException ex)
        {
            CalibrationLoadFailed(_logger, modelId, ex);
            return CreateDefaultCalibrationMap(modelId);
        }
    }

    public async Task<CalibrationMap> FitCalibrationAsync(string modelId, IEnumerable<CalibrationPoint> points, CancellationToken cancellationToken = default)
    {
        try
        {
            var pointsList = points.ToList();
            if (pointsList.Count < MinCalibrationPoints)
            {
                InsufficientDataForCalibration(_logger, modelId, pointsList.Count, null);
                return CreateDefaultCalibrationMap(modelId);
            }

            // Fit both Platt and Isotonic calibration
            var plattMap = FitPlattCalibration(modelId, pointsList);
            var isotonicMap = FitIsotonicCalibration(modelId, pointsList);

            // Choose best method based on Brier score
            var bestMap = plattMap.BrierScore <= isotonicMap.BrierScore ? plattMap : isotonicMap;

            // Save the best calibration map
            await SaveCalibrationMapAsync(bestMap, cancellationToken).ConfigureAwait(false);

            CalibrationFittedWithBrier(_logger, modelId, bestMap.Method.ToString(), bestMap.BrierScore, null);

            return bestMap;
        }
        catch (ArgumentException ex)
        {
            CalibrationFitFailed(_logger, modelId, ex);
            return CreateDefaultCalibrationMap(modelId);
        }
        catch (InvalidOperationException ex)
        {
            CalibrationFitFailed(_logger, modelId, ex);
            return CreateDefaultCalibrationMap(modelId);
        }
        catch (ArithmeticException ex)
        {
            CalibrationFitFailed(_logger, modelId, ex);
            return CreateDefaultCalibrationMap(modelId);
        }
    }

    public async Task<double> CalibrateConfidenceAsync(string modelId, double rawConfidence, CancellationToken cancellationToken = default)
    {
        try
        {
            var map = await LoadCalibrationMapAsync(modelId, cancellationToken).ConfigureAwait(false);
            return ApplyCalibration(map, rawConfidence);
        }
        catch (IOException ex)
        {
            ConfidenceCalibrationFailed(_logger, modelId, ex);
            return rawConfidence; // Return uncalibrated value as fallback
        }
        catch (JsonException ex)
        {
            ConfidenceCalibrationFailed(_logger, modelId, ex);
            return rawConfidence; // Return uncalibrated value as fallback
        }
        catch (ArgumentException ex)
        {
            ConfidenceCalibrationFailed(_logger, modelId, ex);
            return rawConfidence; // Return uncalibrated value as fallback
        }
    }

    public async Task PerformNightlyCalibrationAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            NightlyCalibrationStarted(_logger, DateTime.Now, null);

            var modelsDir = Path.Combine(_basePath, "..", "models", "metadata");
            if (!Directory.Exists(modelsDir))
            {
                ModelsDirectoryNotFound(_logger, modelsDir, null);
                return;
            }

            var modelFiles = Directory.GetFiles(modelsDir, "*.json");
            var updated = 0;

            foreach (var modelFile in modelFiles)
            {
                try
                {
                    var fileName = Path.GetFileNameWithoutExtension(modelFile);
                    var calibrationPoints = await LoadCalibrationPointsAsync(fileName, cancellationToken).ConfigureAwait(false);
                    
                    if (calibrationPoints.Count >= MinStableCalibrationPoints) // Minimum points for stable calibration
                    {
                        await FitCalibrationAsync(fileName, calibrationPoints, cancellationToken).ConfigureAwait(false);
                        updated++;
                    }
                }
                catch (IOException ex)
                {
                    ModelCalibrationUpdateFailed(_logger, modelFile, ex);
                }
                catch (ArgumentException ex)
                {
                    ModelCalibrationUpdateFailed(_logger, modelFile, ex);
                }
                catch (InvalidOperationException ex)
                {
                    ModelCalibrationUpdateFailed(_logger, modelFile, ex);
                }
            }

            NightlyCalibrationCompleted(_logger, updated, null);
        }
        catch (DirectoryNotFoundException ex)
        {
            NightlyCalibrationFailed(_logger, ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            NightlyCalibrationFailed(_logger, ex);
        }
        catch (IOException ex)
        {
            NightlyCalibrationFailed(_logger, ex);
        }
    }

    private static CalibrationMap FitPlattCalibration(string modelId, List<CalibrationPoint> points)
    {
        // Simplified Platt calibration (sigmoid fitting)
        // In production, would use proper logistic regression
        
        var positiveCount = points.Count(p => p.Outcome);
        var totalCount = points.Count;
        var priorPositive = (double)positiveCount / totalCount;
        
        // Simple linear transformation as approximation
        var slope = 1.0;
        var intercept = Math.Log(priorPositive / (1.0 - priorPositive));
        
        var parameters = new Dictionary<string, double>
        {
            ["slope"] = slope,
            ["intercept"] = intercept
        };

        var brierScore = CalculateBrierScore(points, p => ApplyPlattCalibration(p.RawConfidence, slope, intercept));
        var logLoss = CalculateLogLoss(points, p => ApplyPlattCalibration(p.RawConfidence, slope, intercept));

        var calibrationMap = new CalibrationMap
        {
            ModelId = modelId,
            Method = CalibrationMethod.Platt,
            BrierScore = brierScore,
            LogLoss = logLoss,
            CreatedAt = DateTime.UtcNow
        };

        // Add parameters to the dictionary
        foreach (var kvp in parameters)
        {
            calibrationMap.Parameters[kvp.Key] = kvp.Value;
        }

        return calibrationMap;
    }

    private static CalibrationMap FitIsotonicCalibration(string modelId, List<CalibrationPoint> points)
    {
        // Simplified isotonic regression
        // In production, would use proper isotonic regression algorithm
        
        var sortedPoints = points.OrderBy(p => p.RawConfidence).ToList();
        var calibrationBins = new Dictionary<string, double>();
        
        // Create bins for isotonic mapping
        const int numBins = 10;
        var binSize = sortedPoints.Count / numBins;
        
        for (int i = 0; i < numBins; i++)
        {
            var binStart = i * binSize;
            var binEnd = Math.Min((i + 1) * binSize, sortedPoints.Count);
            
            if (binEnd > binStart)
            {
                var binPoints = sortedPoints.Skip(binStart).Take(binEnd - binStart);
                var avgRawConf = binPoints.Average(p => p.RawConfidence);
                var empiricalRate = binPoints.Average(p => p.Outcome ? 1.0 : 0.0);
                
                calibrationBins[$"bin_{i}"] = empiricalRate;
                calibrationBins[$"threshold_{i}"] = avgRawConf;
            }
        }

        var brierScore = CalculateBrierScore(points, p => ApplyIsotonicCalibration(p.RawConfidence, calibrationBins));
        var logLoss = CalculateLogLoss(points, p => ApplyIsotonicCalibration(p.RawConfidence, calibrationBins));

        var calibrationMap = new CalibrationMap
        {
            ModelId = modelId,
            Method = CalibrationMethod.Isotonic,
            BrierScore = brierScore,
            LogLoss = logLoss,
            CreatedAt = DateTime.UtcNow
        };

        // Add parameters to the dictionary
        foreach (var kvp in calibrationBins)
        {
            calibrationMap.Parameters[kvp.Key] = kvp.Value;
        }

        return calibrationMap;
    }

    private static double ApplyCalibration(CalibrationMap map, double rawConfidence)
    {
        return map.Method switch
        {
            CalibrationMethod.Platt => ApplyPlattCalibration(rawConfidence, 
                map.Parameters.GetValueOrDefault("slope", 1.0),
                map.Parameters.GetValueOrDefault("intercept", 0.0)),
            CalibrationMethod.Isotonic => ApplyIsotonicCalibration(rawConfidence, map.Parameters),
            _ => rawConfidence
        };
    }

    private static double ApplyPlattCalibration(double rawConfidence, double slope, double intercept)
    {
        var logit = slope * rawConfidence + intercept;
        return 1.0 / (1.0 + Math.Exp(-logit));
    }

    private static double ApplyIsotonicCalibration(double rawConfidence, Dictionary<string, double> parameters)
    {
        // Find appropriate bin
        var bestBin = 0;
        var minDiff = double.MaxValue;
        
        for (int i = 0; i < DefaultThresholdCount; i++)
        {
            if (parameters.TryGetValue($"threshold_{i}", out var threshold))
            {
                var diff = Math.Abs(rawConfidence - threshold);
                if (diff < minDiff)
                {
                    minDiff = diff;
                    bestBin = i;
                }
            }
        }
        
        return parameters.GetValueOrDefault($"bin_{bestBin}", rawConfidence);
    }

    private static double CalculateBrierScore(List<CalibrationPoint> points, Func<CalibrationPoint, double> predictor)
    {
        if (points.Count == 0) return 1.0;
        
        var sum = 0.0;
        foreach (var point in points)
        {
            var prediction = predictor(point);
            var outcome = point.Outcome ? 1.0 : 0.0;
            sum += Math.Pow(prediction - outcome, ExponentValue) * point.Weight;
        }
        
        return sum / points.Sum(p => p.Weight);
    }

    private static double CalculateLogLoss(List<CalibrationPoint> points, Func<CalibrationPoint, double> predictor)
    {
        if (points.Count == 0) return double.MaxValue;
        
        var sum = 0.0;
        foreach (var point in points)
        {
            var prediction = Math.Max(1e-15, Math.Min(1.0 - 1e-15, predictor(point)));
            var outcome = point.Outcome ? 1.0 : 0.0;
            sum += -(outcome * Math.Log(prediction) + (1.0 - outcome) * Math.Log(1.0 - prediction)) * point.Weight;
        }
        
        return sum / points.Sum(p => p.Weight);
    }

    private static CalibrationMap CreateDefaultCalibrationMap(string modelId)
    {
        var calibrationMap = new CalibrationMap
        {
            ModelId = modelId,
            Method = CalibrationMethod.Platt,
            BrierScore = 0.25, // Worst case for binary classification
            LogLoss = Math.Log(2), // Worst case for binary classification
            CreatedAt = DateTime.UtcNow
        };

        // Set default parameters
        calibrationMap.Parameters["slope"] = 1.0;
        calibrationMap.Parameters["intercept"] = 0.0;

        return calibrationMap;
    }

    private async Task SaveCalibrationMapAsync(CalibrationMap map, CancellationToken cancellationToken)
    {
        var path = Path.Combine(_basePath, $"{map.ModelId}_calibration.json");
        var json = JsonSerializer.Serialize(map, new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(path, json, cancellationToken).ConfigureAwait(false);
        
        lock (_lock)
        {
            _calibrationCache[map.ModelId] = map;
        }
    }

    private async Task<List<CalibrationPoint>> LoadCalibrationPointsAsync(string modelId, CancellationToken cancellationToken)
    {
        var pointsPath = Path.Combine(_basePath, $"{modelId}_points.json");
        if (!File.Exists(pointsPath))
        {
            return new List<CalibrationPoint>();
        }

        try
        {
            var content = await File.ReadAllTextAsync(pointsPath, cancellationToken).ConfigureAwait(false);
            return JsonSerializer.Deserialize<List<CalibrationPoint>>(content) ?? new List<CalibrationPoint>();
        }
        catch (Exception ex)
        {
            CalibrationPointsLoadFailed(_logger, modelId, ex);
            return new List<CalibrationPoint>();
        }
    }

    private void ScheduleNightlyCalibration()
    {
        var now = DateTime.Now;
        var scheduled = new DateTime(now.Year, now.Month, now.Day, 2, 30, 0, DateTimeKind.Local);
        
        // If we've passed today's schedule, schedule for tomorrow
        if (scheduled <= now)
        {
            scheduled = scheduled.AddDays(1);
        }

        var delay = scheduled - now;
        
        _nightlyTimer = new Timer(_ =>
        {
            _ = Task.Run(async () =>
            {
                try
                {
                    await PerformNightlyCalibrationAsync(CancellationToken.None).ConfigureAwait(false);
                }
                catch (Exception ex)
                {
                    NightlyCalibrationFailed(_logger, ex);
                }
            });
            
            // Reschedule for next day
            _nightlyTimer?.Change(TimeSpan.FromDays(1), Timeout.InfiniteTimeSpan);
            
        }, null, delay, Timeout.InfiniteTimeSpan);

        NightlyCalibrationScheduled(_logger, scheduled, null);
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }
    
    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            _nightlyTimer?.Dispose();
        }
    }
}