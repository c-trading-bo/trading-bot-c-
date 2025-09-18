using System;
using System.Collections.Generic;
using System.Data.SQLite;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace UnifiedOrchestrator.Services
{
    public class DataLakeOptions
    {
        public string DatabasePath { get; set; } = "data/feature_lake.db";
        public double DriftThreshold { get; set; } = 0.05;
        public bool AlertOnDrift { get; set; } = true;
        public int MaxRetentionDays { get; set; } = 365;
    }

    public class FeatureSet
    {
        public string FeatureSetName { get; set; } = "";
        public Dictionary<string, object> Features { get; } = new();
        public DateTime Timestamp { get; set; }
        public string InstanceId { get; set; } = "";
    }

    public class DataQualityReport
    {
        public string FeatureSetName { get; set; } = "";
        public DateTime GeneratedAt { get; set; }
        public int TotalRecords { get; set; }
        public double CompletenessScore { get; set; }
        public double DriftScore { get; set; }
        public bool IsHealthy { get; set; }
        public List<string> Issues { get; } = new();
    }

    public class DataLakeService : IDisposable
    {
        private readonly ILogger<DataLakeService> _logger;
        private readonly DataLakeOptions _options;
        private SQLiteConnection? _connection;
        private readonly object _connectionLock = new object();

        public DataLakeService(ILogger<DataLakeService> logger, IOptions<DataLakeOptions> options)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
            
            InitializeDatabase();
        }

        private void InitializeDatabase()
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(_options.DatabasePath) ?? "data");
                
                lock (_connectionLock)
                {
                    _connection = new SQLiteConnection($"Data Source={_options.DatabasePath}");
                    _connection.Open();

                    var createTableSql = @"
                        CREATE TABLE IF NOT EXISTS feature_sets (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            feature_set_name TEXT NOT NULL,
                            features TEXT NOT NULL,
                            timestamp TEXT NOT NULL,
                            instance_id TEXT,
                            created_at TEXT DEFAULT CURRENT_TIMESTAMP
                        );
                        
                        CREATE INDEX IF NOT EXISTS idx_feature_sets_name_timestamp 
                        ON feature_sets(feature_set_name, timestamp);
                    ";

                    var command = new SQLiteCommand(createTableSql, _connection);
                    command.ExecuteNonQuery();
                }

                _logger.LogInformation("Initialized DataLake database at {DatabasePath}", _options.DatabasePath);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to initialize DataLake database");
                throw;
            }
        }

        public async Task<bool> StoreFeatureSetAsync(string featureSetName, Dictionary<string, object> features, DateTime timestamp, CancellationToken cancellationToken)
        {
            try
            {
                var featuresJson = JsonSerializer.Serialize(features);
                var timestampString = timestamp.ToString("O");

                // Check for drift if enabled
                if (_options.AlertOnDrift)
                {
                    var driftScore = await CalculateDriftScoreAsync(featureSetName, features, cancellationToken).ConfigureAwait(false);
                    if (driftScore > _options.DriftThreshold)
                    {
                        _logger.LogWarning("Feature drift detected for {FeatureSetName}: {DriftScore:F3} > {Threshold:F3}", 
                            featureSetName, driftScore, _options.DriftThreshold);
                    }
                }

                lock (_connectionLock)
                {
                    if (_connection == null)
                        throw new InvalidOperationException("Database connection not initialized");

                    var insertSql = @"
                        INSERT INTO feature_sets (feature_set_name, features, timestamp, instance_id)
                        VALUES (@featureSetName, @features, @timestamp, @instanceId)
                    ";

                    var command = new SQLiteCommand(insertSql, _connection);
                    command.Parameters.AddWithValue("@featureSetName", featureSetName);
                    command.Parameters.AddWithValue("@features", featuresJson);
                    command.Parameters.AddWithValue("@timestamp", timestampString);
                    command.Parameters.AddWithValue("@instanceId", Environment.MachineName);

                    var rowsAffected = command.ExecuteNonQuery();
                    
                    _logger.LogDebug("Stored feature set {FeatureSetName} with {FeatureCount} features", 
                        featureSetName, features.Count);
                    
                    return rowsAffected > 0;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to store feature set {FeatureSetName}", featureSetName);
                return false;
            }
        }

        public Task<List<FeatureSet>> RetrieveFeaturesAsync(string featureSetName, DateTime startTime, DateTime endTime, CancellationToken cancellationToken)
        {
            try
            {
                var results = new List<FeatureSet>();

                lock (_connectionLock)
                {
                    if (_connection == null)
                        throw new InvalidOperationException("Database connection not initialized");

                    var selectSql = @"
                        SELECT feature_set_name, features, timestamp, instance_id
                        FROM feature_sets
                        WHERE feature_set_name = @featureSetName
                        AND timestamp >= @startTime
                        AND timestamp <= @endTime
                        ORDER BY timestamp
                    ";

                    var command = new SQLiteCommand(selectSql, _connection);
                    command.Parameters.AddWithValue("@featureSetName", featureSetName);
                    command.Parameters.AddWithValue("@startTime", startTime.ToString("O"));
                    command.Parameters.AddWithValue("@endTime", endTime.ToString("O"));

                    using var reader = command.ExecuteReader();
                    while (reader.Read())
                    {
                        var featuresJson = reader.GetString(1); // features column
                        var features = JsonSerializer.Deserialize<Dictionary<string, object>>(featuresJson) ?? new();
                        
                        results.Add(new FeatureSet
                        {
                            FeatureSetName = reader.GetString(0), // feature_set_name column
                            Features = features,
                            Timestamp = DateTime.Parse(reader.GetString(2)), // timestamp column
                            InstanceId = reader.IsDBNull(3) ? "" : reader.GetString(3) // instance_id column
                        });
                    }
                }

                _logger.LogDebug("Retrieved {Count} feature sets for {FeatureSetName}", results.Count, featureSetName);
                return Task.FromResult(results);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to retrieve features for {FeatureSetName}", featureSetName);
                return Task.FromResult(new List<FeatureSet>());
            }
        }

        public async Task<DataQualityReport> GetDataQualityReportAsync(string featureSetName, TimeSpan lookbackWindow, CancellationToken cancellationToken)
        {
            try
            {
                var endTime = DateTime.UtcNow;
                var startTime = endTime - lookbackWindow;
                
                var featureSets = await RetrieveFeaturesAsync(featureSetName, startTime, endTime, cancellationToken).ConfigureAwait(false);
                
                var report = new DataQualityReport
                {
                    FeatureSetName = featureSetName,
                    GeneratedAt = DateTime.UtcNow,
                    TotalRecords = featureSets.Count,
                    CompletenessScore = CalculateCompletenessScore(featureSets),
                    DriftScore = await CalculateAverageDriftScoreAsync(featureSetName, featureSets, cancellationToken),
                    Issues = new List<string>()
                }.ConfigureAwait(false);

                report.IsHealthy = report.CompletenessScore > 0.95 && report.DriftScore < _options.DriftThreshold;

                if (report.CompletenessScore <= 0.95)
                    report.Issues.Add($"Low completeness score: {report.CompletenessScore:F2}");
                
                if (report.DriftScore >= _options.DriftThreshold)
                    report.Issues.Add($"High drift score: {report.DriftScore:F3}");

                return report;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to generate data quality report for {FeatureSetName}", featureSetName);
                throw;
            }
        }

        private async Task<double> CalculateDriftScoreAsync(string featureSetName, Dictionary<string, object> currentFeatures, CancellationToken cancellationToken)
        {
            try
            {
                // Get recent historical features for comparison
                var endTime = DateTime.UtcNow.AddMinutes(-1);
                var startTime = endTime.AddHours(-1);
                
                var historicalFeatures = await RetrieveFeaturesAsync(featureSetName, startTime, endTime, cancellationToken).ConfigureAwait(false);
                
                if (historicalFeatures.Count == 0)
                    return 0.0; // No historical data to compare

                // Simple statistical drift detection
                double totalDrift = 0.0;
                int numericFeatureCount = 0;

                foreach (var feature in currentFeatures)
                {
                    if (feature.Value is double || feature.Value is float || feature.Value is decimal)
                    {
                        var currentValue = Convert.ToDouble(feature.Value);
                        var historicalValues = new List<double>();

                        foreach (var historical in historicalFeatures)
                        {
                            if (historical.Features.TryGetValue(feature.Key, out var historicalValue))
                            {
                                historicalValues.Add(Convert.ToDouble(historicalValue));
                            }
                        }

                        if (historicalValues.Count > 0)
                        {
                            var mean = historicalValues.Sum() / historicalValues.Count;
                            var variance = historicalValues.Sum(x => Math.Pow(x - mean, 2)) / historicalValues.Count;
                            var stdDev = Math.Sqrt(variance);

                            if (stdDev > 0)
                            {
                                var zScore = Math.Abs((currentValue - mean) / stdDev);
                                totalDrift += Math.Min(zScore / 3.0, 1.0); // Normalize to [0,1]
                                numericFeatureCount++;
                            }
                        }
                    }
                }

                return numericFeatureCount > 0 ? totalDrift / numericFeatureCount : 0.0;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to calculate drift score for {FeatureSetName}", featureSetName);
                return 0.0;
            }
        }

        private double CalculateCompletenessScore(List<FeatureSet> featureSets)
        {
            if (featureSets.Count == 0)
                return 0.0;

            var allFeatureNames = new HashSet<string>();
            foreach (var featureSet in featureSets)
            {
                foreach (var feature in featureSet.Features.Keys)
                {
                    allFeatureNames.Add(feature);
                }
            }

            if (allFeatureNames.Count == 0)
                return 1.0;

            double totalCompleteness = 0.0;
            foreach (var featureSet in featureSets)
            {
                var presentFeatures = featureSet.Features.Keys.Count;
                var completeness = (double)presentFeatures / allFeatureNames.Count;
                totalCompleteness += completeness;
            }

            return totalCompleteness / featureSets.Count;
        }

        private async Task<double> CalculateAverageDriftScoreAsync(string featureSetName, List<FeatureSet> featureSets, CancellationToken cancellationToken)
        {
            if (featureSets.Count <= 1)
                return 0.0;

            double totalDrift = 0.0;
            int comparisons = 0;

            for (int i = 1; i < featureSets.Count; i++)
            {
                var drift = await CalculateDriftScoreAsync(featureSetName, featureSets[i].Features, cancellationToken).ConfigureAwait(false);
                totalDrift += drift;
                comparisons++;
            }

            return comparisons > 0 ? totalDrift / comparisons : 0.0;
        }

        public void Dispose()
        {
            lock (_connectionLock)
            {
                _connection?.Close();
                _connection?.Dispose();
                _connection = null!;
            }
        }
    }
}