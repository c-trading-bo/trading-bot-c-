using System;
using System.Collections.Generic;
using System.Data.SQLite;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace UnifiedOrchestrator.Services
{
    /// <summary>
    /// Data Lake and Feature Store with schema validation and drift detection
    /// Uses SQLite as lightweight time-series database for backup, replay, and data quality
    /// </summary>
    public class DataLakeService : IDisposable
    {
        private readonly ILogger<DataLakeService> _logger;
        private readonly DataLakeOptions _options;
        private readonly JsonSerializerOptions _jsonOptions;
        private SQLiteConnection? _connection;
        private readonly Dictionary<string, FeatureSchema> _schemas = new();
        private readonly Dictionary<string, FeatureDriftDetector> _driftDetectors = new();
        private bool _disposed = false;

        public DataLakeService(
            ILogger<DataLakeService> logger,
            IOptions<DataLakeOptions> options)
        {
            _logger = logger;
            _options = options.Value;
            
            _jsonOptions = new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                WriteIndented = false
            };

            InitializeDatabaseAsync().GetAwaiter().GetResult();
            _logger.LogInformation("🏞️ Data Lake Service initialized: {DatabasePath}", _options.DatabasePath);
        }

        /// <summary>
        /// Store features with schema validation and drift detection
        /// </summary>
        public async Task<bool> StoreFeatureSetAsync(
            string featureSetName,
            Dictionary<string, object> features,
            DateTime timestamp,
            CancellationToken cancellationToken = default)
        {
            try
            {
                // Validate schema
                var isValidSchema = await ValidateSchemaAsync(featureSetName, features, cancellationToken);
                if (!isValidSchema)
                {
                    _logger.LogWarning("⚠️ Schema validation failed for feature set: {FeatureSetName}", featureSetName);
                    return false;
                }

                // Detect drift
                var driftScore = await DetectDriftAsync(featureSetName, features, cancellationToken);
                if (driftScore > _options.DriftThreshold)
                {
                    _logger.LogWarning("📈 Feature drift detected: {FeatureSetName}, Score: {DriftScore:F4}", 
                        featureSetName, driftScore);
                    
                    if (_options.AlertOnDrift)
                    {
                        await TriggerDriftAlertAsync(featureSetName, driftScore, cancellationToken);
                    }
                }

                // Store in database
                await StoreInDatabaseAsync(featureSetName, features, timestamp, driftScore, cancellationToken);
                
                _logger.LogDebug("💾 Feature set stored: {FeatureSetName} at {Timestamp}", featureSetName, timestamp);
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to store feature set: {FeatureSetName}", featureSetName);
                return false;
            }
        }

        /// <summary>
        /// Retrieve features for replay
        /// </summary>
        public async Task<List<FeatureRecord>> RetrieveFeaturesAsync(
            string featureSetName,
            DateTime startTime,
            DateTime endTime,
            CancellationToken cancellationToken = default)
        {
            var features = new List<FeatureRecord>();

            try
            {
                const string query = @"
                    SELECT timestamp, features_json, drift_score 
                    FROM feature_store 
                    WHERE feature_set_name = @featureSetName 
                      AND timestamp BETWEEN @startTime AND @endTime 
                    ORDER BY timestamp";

                using var command = new SQLiteCommand(query, _connection);
                command.Parameters.AddWithValue("@featureSetName", featureSetName);
                command.Parameters.AddWithValue("@startTime", startTime.ToString("yyyy-MM-dd HH:mm:ss.fff"));
                command.Parameters.AddWithValue("@endTime", endTime.ToString("yyyy-MM-dd HH:mm:ss.fff"));

                using var reader = await command.ExecuteReaderAsync(cancellationToken);
                while (await reader.ReadAsync(cancellationToken))
                {
                    var timestampStr = reader.GetString(0);
                    var featuresJson = reader.GetString(1);
                    var driftScore = reader.GetDouble(2);

                    if (DateTime.TryParse(timestampStr, out var timestamp))
                    {
                        var featuresDict = JsonSerializer.Deserialize<Dictionary<string, object>>(featuresJson, _jsonOptions);
                        features.Add(new FeatureRecord
                        {
                            FeatureSetName = featureSetName,
                            Timestamp = timestamp,
                            Features = featuresDict ?? new(),
                            DriftScore = driftScore
                        });
                    }
                }

                _logger.LogDebug("📊 Retrieved {Count} feature records for {FeatureSetName}", features.Count, featureSetName);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to retrieve features: {FeatureSetName}", featureSetName);
            }

            return features;
        }

        /// <summary>
        /// Register or update feature schema
        /// </summary>
        public async Task RegisterSchemaAsync(string featureSetName, FeatureSchema schema, CancellationToken cancellationToken = default)
        {
            try
            {
                _schemas[featureSetName] = schema;
                
                // Store schema in database
                const string query = @"
                    INSERT OR REPLACE INTO feature_schemas (feature_set_name, schema_json, created_at)
                    VALUES (@featureSetName, @schemaJson, @createdAt)";

                var schemaJson = JsonSerializer.Serialize(schema, _jsonOptions);
                using var command = new SQLiteCommand(query, _connection);
                command.Parameters.AddWithValue("@featureSetName", featureSetName);
                command.Parameters.AddWithValue("@schemaJson", schemaJson);
                command.Parameters.AddWithValue("@createdAt", DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss.fff"));

                await command.ExecuteNonQueryAsync(cancellationToken);
                _logger.LogInformation("📋 Schema registered for feature set: {FeatureSetName}", featureSetName);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to register schema: {FeatureSetName}", featureSetName);
                throw;
            }
        }

        /// <summary>
        /// Get data quality report
        /// </summary>
        public async Task<DataQualityReport> GetDataQualityReportAsync(
            string featureSetName,
            TimeSpan timeWindow,
            CancellationToken cancellationToken = default)
        {
            var report = new DataQualityReport
            {
                FeatureSetName = featureSetName,
                TimeWindow = timeWindow,
                Timestamp = DateTime.UtcNow
            };

            try
            {
                var endTime = DateTime.UtcNow;
                var startTime = endTime - timeWindow;

                // Get feature records
                var features = await RetrieveFeaturesAsync(featureSetName, startTime, endTime, cancellationToken);
                
                report.TotalRecords = features.Count;
                report.AverageDriftScore = features.Count > 0 ? features.Average(f => f.DriftScore) : 0;
                report.MaxDriftScore = features.Count > 0 ? features.Max(f => f.DriftScore) : 0;
                report.DriftAlertsCount = features.Count(f => f.DriftScore > _options.DriftThreshold);

                // Check for missing data gaps
                if (features.Count > 1)
                {
                    var intervals = new List<TimeSpan>();
                    for (int i = 1; i < features.Count; i++)
                    {
                        intervals.Add(features[i].Timestamp - features[i-1].Timestamp);
                    }
                    
                    var averageInterval = TimeSpan.FromTicks((long)intervals.Average(i => i.Ticks));
                    var maxGap = intervals.Max();
                    
                    report.AverageDataInterval = averageInterval;
                    report.MaxDataGap = maxGap;
                    report.HasDataGaps = maxGap > averageInterval.Multiply(2); // Flag if gap > 2x average
                }

                report.IsHealthy = report.DriftAlertsCount == 0 && !report.HasDataGaps;
                _logger.LogDebug("📊 Data quality report generated for {FeatureSetName}: {HealthStatus}", 
                    featureSetName, report.IsHealthy ? "Healthy" : "Issues detected");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to generate data quality report: {FeatureSetName}", featureSetName);
                report.IsHealthy = false;
            }

            return report;
        }

        /// <summary>
        /// Check if trading should be paused due to stale or missing data
        /// </summary>
        public async Task<bool> ShouldPauseTradingAsync(CancellationToken cancellationToken = default)
        {
            try
            {
                var cutoffTime = DateTime.UtcNow - TimeSpan.FromMinutes(_options.StaleDataThresholdMinutes);
                
                const string query = @"
                    SELECT feature_set_name, MAX(timestamp) as last_update
                    FROM feature_store 
                    GROUP BY feature_set_name";

                using var command = new SQLiteCommand(query, _connection);
                using var reader = await command.ExecuteReaderAsync(cancellationToken);

                var staleFeatureSets = new List<string>();
                while (await reader.ReadAsync(cancellationToken))
                {
                    var featureSetName = reader.GetString(0);
                    var lastUpdateStr = reader.GetString(1);
                    
                    if (DateTime.TryParse(lastUpdateStr, out var lastUpdate) && lastUpdate < cutoffTime)
                    {
                        staleFeatureSets.Add(featureSetName);
                    }
                }

                if (staleFeatureSets.Any())
                {
                    _logger.LogWarning("⚠️ Stale data detected in feature sets: {StaleSets}", string.Join(", ", staleFeatureSets));
                    return true;
                }

                return false;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to check for stale data");
                return true; // Err on the side of caution
            }
        }

        private async Task InitializeDatabaseAsync()
        {
            try
            {
                var dbDir = Path.GetDirectoryName(_options.DatabasePath);
                if (!string.IsNullOrEmpty(dbDir) && !Directory.Exists(dbDir))
                {
                    Directory.CreateDirectory(dbDir);
                }

                _connection = new SQLiteConnection($"Data Source={_options.DatabasePath}");
                await _connection.OpenAsync();

                // Create tables
                await CreateTablesAsync();
                
                // Load existing schemas
                await LoadSchemasAsync();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to initialize database");
                throw;
            }
        }

        private async Task CreateTablesAsync()
        {
            var createTablesScript = @"
                CREATE TABLE IF NOT EXISTS feature_store (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_set_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    drift_score REAL NOT NULL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS feature_schemas (
                    feature_set_name TEXT PRIMARY KEY,
                    schema_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_feature_store_name_time 
                ON feature_store(feature_set_name, timestamp);

                CREATE INDEX IF NOT EXISTS idx_feature_store_timestamp 
                ON feature_store(timestamp);
            ";

            using var command = new SQLiteCommand(createTablesScript, _connection);
            await command.ExecuteNonQueryAsync();
        }

        private async Task LoadSchemasAsync()
        {
            const string query = "SELECT feature_set_name, schema_json FROM feature_schemas";
            using var command = new SQLiteCommand(query, _connection);
            using var reader = await command.ExecuteReaderAsync();

            while (await reader.ReadAsync())
            {
                var featureSetName = reader.GetString(0);
                var schemaJson = reader.GetString(1);
                
                var schema = JsonSerializer.Deserialize<FeatureSchema>(schemaJson, _jsonOptions);
                if (schema != null)
                {
                    _schemas[featureSetName] = schema;
                }
            }
        }

        private async Task<bool> ValidateSchemaAsync(string featureSetName, Dictionary<string, object> features, CancellationToken cancellationToken)
        {
            if (!_schemas.TryGetValue(featureSetName, out var schema))
            {
                // Auto-register schema if not exists
                var autoSchema = new FeatureSchema
                {
                    Name = featureSetName,
                    Fields = features.ToDictionary(
                        kvp => kvp.Key, 
                        kvp => new FeatureField { Name = kvp.Key, Type = InferType(kvp.Value), Required = false }
                    )
                };
                await RegisterSchemaAsync(featureSetName, autoSchema, cancellationToken);
                return true;
            }

            // Validate required fields
            foreach (var field in schema.Fields.Values.Where(f => f.Required))
            {
                if (!features.ContainsKey(field.Name))
                {
                    _logger.LogWarning("❌ Missing required field: {FieldName} in {FeatureSetName}", field.Name, featureSetName);
                    return false;
                }
            }

            // Validate types (basic validation)
            foreach (var kvp in features)
            {
                if (schema.Fields.TryGetValue(kvp.Key, out var field))
                {
                    var actualType = InferType(kvp.Value);
                    if (actualType != field.Type && field.Type != "object")
                    {
                        _logger.LogWarning("⚠️ Type mismatch for field {FieldName}: expected {ExpectedType}, got {ActualType}",
                            kvp.Key, field.Type, actualType);
                    }
                }
            }

            return true;
        }

        private async Task<double> DetectDriftAsync(string featureSetName, Dictionary<string, object> currentFeatures, CancellationToken cancellationToken)
        {
            if (!_driftDetectors.TryGetValue(featureSetName, out var detector))
            {
                detector = new FeatureDriftDetector(featureSetName, _options.DriftWindowSize);
                _driftDetectors[featureSetName] = detector;
            }

            return await detector.AddSampleAndCalculateDriftAsync(currentFeatures, cancellationToken);
        }

        private async Task TriggerDriftAlertAsync(string featureSetName, double driftScore, CancellationToken cancellationToken)
        {
            // Simple logging alert - could be enhanced with external alerting
            _logger.LogWarning("🚨 DRIFT ALERT: {FeatureSetName} - Score: {DriftScore:F4}", featureSetName, driftScore);
            
            // Could integrate with external alerting systems here
            await Task.CompletedTask;
        }

        private async Task StoreInDatabaseAsync(string featureSetName, Dictionary<string, object> features, DateTime timestamp, double driftScore, CancellationToken cancellationToken)
        {
            const string query = @"
                INSERT INTO feature_store (feature_set_name, timestamp, features_json, drift_score)
                VALUES (@featureSetName, @timestamp, @featuresJson, @driftScore)";

            var featuresJson = JsonSerializer.Serialize(features, _jsonOptions);
            using var command = new SQLiteCommand(query, _connection);
            command.Parameters.AddWithValue("@featureSetName", featureSetName);
            command.Parameters.AddWithValue("@timestamp", timestamp.ToString("yyyy-MM-dd HH:mm:ss.fff"));
            command.Parameters.AddWithValue("@featuresJson", featuresJson);
            command.Parameters.AddWithValue("@driftScore", driftScore);

            await command.ExecuteNonQueryAsync(cancellationToken);
        }

        private static string InferType(object value)
        {
            return value switch
            {
                int or long => "integer",
                float or double or decimal => "number",
                bool => "boolean",
                string => "string",
                DateTime => "datetime",
                _ => "object"
            };
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _connection?.Close();
                _connection?.Dispose();
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Configuration options for data lake service
    /// </summary>
    public class DataLakeOptions
    {
        public string DatabasePath { get; set; } = "data/feature_store.db";
        public double DriftThreshold { get; set; } = 0.05;
        public bool AlertOnDrift { get; set; } = true;
        public int DriftWindowSize { get; set; } = 1000;
        public int StaleDataThresholdMinutes { get; set; } = 5;
    }

    /// <summary>
    /// Feature schema definition
    /// </summary>
    public class FeatureSchema
    {
        public string Name { get; set; } = string.Empty;
        public Dictionary<string, FeatureField> Fields { get; set; } = new();
    }

    /// <summary>
    /// Feature field definition
    /// </summary>
    public class FeatureField
    {
        public string Name { get; set; } = string.Empty;
        public string Type { get; set; } = string.Empty;
        public bool Required { get; set; }
        public object? DefaultValue { get; set; }
    }

    /// <summary>
    /// Feature record for storage and retrieval
    /// </summary>
    public class FeatureRecord
    {
        public string FeatureSetName { get; set; } = string.Empty;
        public DateTime Timestamp { get; set; }
        public Dictionary<string, object> Features { get; set; } = new();
        public double DriftScore { get; set; }
    }

    /// <summary>
    /// Data quality report
    /// </summary>
    public class DataQualityReport
    {
        public string FeatureSetName { get; set; } = string.Empty;
        public DateTime Timestamp { get; set; }
        public TimeSpan TimeWindow { get; set; }
        public int TotalRecords { get; set; }
        public double AverageDriftScore { get; set; }
        public double MaxDriftScore { get; set; }
        public int DriftAlertsCount { get; set; }
        public TimeSpan AverageDataInterval { get; set; }
        public TimeSpan MaxDataGap { get; set; }
        public bool HasDataGaps { get; set; }
        public bool IsHealthy { get; set; }
    }

    /// <summary>
    /// Feature drift detector using statistical methods
    /// </summary>
    public class FeatureDriftDetector
    {
        private readonly string _featureSetName;
        private readonly int _windowSize;
        private readonly Queue<Dictionary<string, object>> _referenceWindow = new();
        private readonly Dictionary<string, (double sum, double sumSquared, int count)> _referenceStats = new();

        public FeatureDriftDetector(string featureSetName, int windowSize)
        {
            _featureSetName = featureSetName;
            _windowSize = windowSize;
        }

        public async Task<double> AddSampleAndCalculateDriftAsync(Dictionary<string, object> sample, CancellationToken cancellationToken)
        {
            await Task.Yield(); // Make it async

            // Add to reference window
            _referenceWindow.Enqueue(sample);
            if (_referenceWindow.Count > _windowSize)
            {
                var oldest = _referenceWindow.Dequeue();
                UpdateStatsRemove(oldest);
            }
            UpdateStatsAdd(sample);

            // Calculate drift score (simplified KL-divergence approximation)
            return CalculateDriftScore(sample);
        }

        private void UpdateStatsAdd(Dictionary<string, object> sample)
        {
            foreach (var kvp in sample)
            {
                if (kvp.Value is double or float or int or long or decimal)
                {
                    var value = Convert.ToDouble(kvp.Value);
                    if (_referenceStats.TryGetValue(kvp.Key, out var stats))
                    {
                        _referenceStats[kvp.Key] = (stats.sum + value, stats.sumSquared + value * value, stats.count + 1);
                    }
                    else
                    {
                        _referenceStats[kvp.Key] = (value, value * value, 1);
                    }
                }
            }
        }

        private void UpdateStatsRemove(Dictionary<string, object> sample)
        {
            foreach (var kvp in sample)
            {
                if (kvp.Value is double or float or int or long or decimal && _referenceStats.TryGetValue(kvp.Key, out var stats))
                {
                    var value = Convert.ToDouble(kvp.Value);
                    _referenceStats[kvp.Key] = (stats.sum - value, stats.sumSquared - value * value, stats.count - 1);
                }
            }
        }

        private double CalculateDriftScore(Dictionary<string, object> sample)
        {
            if (_referenceStats.Count == 0)
                return 0.0;

            double totalDrift = 0.0;
            int numericFeatures = 0;

            foreach (var kvp in sample)
            {
                if (kvp.Value is double or float or int or long or decimal && _referenceStats.TryGetValue(kvp.Key, out var stats))
                {
                    if (stats.count > 1)
                    {
                        var currentValue = Convert.ToDouble(kvp.Value);
                        var referenceMean = stats.sum / stats.count;
                        var referenceVariance = (stats.sumSquared / stats.count) - (referenceMean * referenceMean);
                        var referenceStd = Math.Sqrt(Math.Max(referenceVariance, 1e-10));

                        // Z-score based drift
                        var zScore = Math.Abs(currentValue - referenceMean) / referenceStd;
                        totalDrift += Math.Min(zScore, 10.0); // Cap extreme values
                        numericFeatures++;
                    }
                }
            }

            return numericFeatures > 0 ? totalDrift / numericFeatures : 0.0;
        }
    }
}