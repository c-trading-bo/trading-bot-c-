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
                _logger.LogWarning("[CALIBRATION] No calibration map found for model: {ModelId}", modelId);
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
                
                _logger.LogDebug("[CALIBRATION] Loaded calibration map for {ModelId} (method: {Method})", 
                    modelId, map.Method);
                return map;
            }

            return CreateDefaultCalibrationMap(modelId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[CALIBRATION] Failed to load calibration map for {ModelId}", modelId);
            return CreateDefaultCalibrationMap(modelId);
        }
    }

    public async Task<CalibrationMap> FitCalibrationAsync(string modelId, IEnumerable<CalibrationPoint> points, CancellationToken cancellationToken = default)
    {
        try
        {
            var pointsList = points.ToList();
            if (pointsList.Count < 10)
            {
                _logger.LogWarning("[CALIBRATION] Insufficient data for calibration: {ModelId} ({Count} points)", 
                    modelId, pointsList.Count);
                return CreateDefaultCalibrationMap(modelId);
            }

            // Fit both Platt and Isotonic calibration
            var plattMap = FitPlattCalibration(modelId, pointsList);
            var isotonicMap = FitIsotonicCalibration(modelId, pointsList);

            // Choose best method based on Brier score
            var bestMap = plattMap.BrierScore <= isotonicMap.BrierScore ? plattMap : isotonicMap;

            // Save the best calibration map
            await SaveCalibrationMapAsync(bestMap, cancellationToken).ConfigureAwait(false);

            _logger.LogInformation("[CALIBRATION] Fitted calibration for {ModelId}: {Method} (Brier: {Brier:F4})", 
                modelId, bestMap.Method, bestMap.BrierScore);

            return bestMap;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[CALIBRATION] Failed to fit calibration for {ModelId}", modelId);
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
        catch (Exception ex)
        {
            _logger.LogError(ex, "[CALIBRATION] Failed to calibrate confidence for {ModelId}", modelId);
            return rawConfidence; // Return uncalibrated value as fallback
        }
    }

    public async Task PerformNightlyCalibrationAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogInformation("[CALIBRATION] Starting nightly calibration update at {Time}", DateTime.Now);

            var modelsDir = Path.Combine(_basePath, "..", "models", "metadata");
            if (!Directory.Exists(modelsDir))
            {
                _logger.LogWarning("[CALIBRATION] Models directory not found: {Dir}", modelsDir);
                return;
            }

            var modelFiles = Directory.GetFiles(modelsDir, "*.json");
            var updated;

            foreach (var modelFile in modelFiles)
            {
                try
                {
                    var fileName = Path.GetFileNameWithoutExtension(modelFile);
                    var calibrationPoints = await LoadCalibrationPointsAsync(fileName, cancellationToken).ConfigureAwait(false);
                    
                    if (calibrationPoints.Count >= 50) // Minimum points for stable calibration
                    {
                        await FitCalibrationAsync(fileName, calibrationPoints, cancellationToken).ConfigureAwait(false);
                        updated++;
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "[CALIBRATION] Failed to update calibration for model: {Model}", modelFile);
                }
            }

            _logger.LogInformation("[CALIBRATION] Nightly calibration completed: {Updated} models updated", updated);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[CALIBRATION] Nightly calibration failed");
        }
    }

    private CalibrationMap FitPlattCalibration(string modelId, List<CalibrationPoint> points)
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

        return new CalibrationMap
        {
            ModelId = modelId,
            Method = CalibrationMethod.Platt,
            Parameters = parameters,
            BrierScore = brierScore,
            LogLoss = logLoss,
            CreatedAt = DateTime.UtcNow
        };
    }

    private CalibrationMap FitIsotonicCalibration(string modelId, List<CalibrationPoint> points)
    {
        // Simplified isotonic regression
        // In production, would use proper isotonic regression algorithm
        
        var sortedPoints = points.OrderBy(p => p.RawConfidence).ToList();
        var calibrationBins = new Dictionary<string, double>();
        
        // Create bins for isotonic mapping
        const int numBins = 10;
        var binSize = sortedPoints.Count / numBins;
        
        for (int i; i < numBins; i++)
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

        return new CalibrationMap
        {
            ModelId = modelId,
            Method = CalibrationMethod.Isotonic,
            Parameters = calibrationBins,
            BrierScore = brierScore,
            LogLoss = logLoss,
            CreatedAt = DateTime.UtcNow
        };
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
        var bestBin;
        var minDiff = double.MaxValue;
        
        for (int i; i < 10; i++)
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
            sum += Math.Pow(prediction - outcome, 2) * point.Weight;
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

    private CalibrationMap CreateDefaultCalibrationMap(string modelId)
    {
        return new CalibrationMap
        {
            ModelId = modelId,
            Method = CalibrationMethod.Platt,
            Parameters = new Dictionary<string, double> { ["slope"] = 1.0, ["intercept"] = 0.0 },
            BrierScore = 0.25, // Worst case for binary classification
            LogLoss = Math.Log(2), // Worst case for binary classification
            CreatedAt = DateTime.UtcNow
        };
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
            _logger.LogWarning(ex, "[CALIBRATION] Failed to load calibration points for {ModelId}", modelId);
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
                    _logger.LogError(ex, "[CALIBRATION] Nightly calibration failed");
                }
            });
            
            // Reschedule for next day
            _nightlyTimer?.Change(TimeSpan.FromDays(1), Timeout.InfiniteTimeSpan);
            
        }, null, delay, Timeout.InfiniteTimeSpan);

        _logger.LogInformation("[CALIBRATION] Scheduled nightly calibration for {Time}", scheduled);
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