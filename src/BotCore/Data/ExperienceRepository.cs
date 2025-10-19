using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using BotCore.Models;
using Microsoft.Extensions.Logging;

namespace BotCore.Data
{
    /// <summary>
    /// Repository for storing and retrieving trading experiences
    /// Task 4.1: Experience Collection in Terminal
    /// 
    /// Stores experiences as JSON files for Lab training pipeline
    /// Terminal writes, Lab reads (simple file-based handoff)
    /// </summary>
    public sealed class ExperienceRepository
    {
        private readonly ILogger<ExperienceRepository> _logger;
        private readonly string _experienceDirectory;
        private static readonly JsonSerializerOptions _jsonOptions = new()
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        public ExperienceRepository(ILogger<ExperienceRepository> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            
            // Store experiences in data/experiences/ directory
            _experienceDirectory = Path.Combine(
                Directory.GetCurrentDirectory(),
                "data",
                "experiences"
            );
            
            // Create directory if it doesn't exist
            Directory.CreateDirectory(_experienceDirectory);
            
            _logger.LogInformation("📁 [EXPERIENCE-REPO] Initialized: {Directory}", _experienceDirectory);
        }

        /// <summary>
        /// Save a trading experience to disk
        /// Called by Terminal when positions close
        /// </summary>
        public async Task SaveExperienceAsync(TradingExperience experience)
        {
            if (experience == null)
            {
                throw new ArgumentNullException(nameof(experience));
            }

            try
            {
                // Generate filename with timestamp and experience ID
                var timestamp = experience.Timestamp.ToString("yyyy-MM-dd_HHmmss");
                var filename = $"{timestamp}_{experience.ExperienceId}.json";
                var filepath = Path.Combine(_experienceDirectory, filename);

                // Serialize to JSON
                var json = JsonSerializer.Serialize(experience, _jsonOptions);

                // Write to file (atomic operation)
                await File.WriteAllTextAsync(filepath, json).ConfigureAwait(false);

                _logger.LogInformation(
                    "💾 [EXPERIENCE-REPO] Saved: {Strategy} - R: {RMultiple:F2}, PnL: ${PnL:F2}, Duration: {Duration:F1}m",
                    experience.Strategy,
                    experience.RMultiple,
                    experience.PnL,
                    experience.DurationMinutes
                );
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [EXPERIENCE-REPO] Failed to save experience: {ExperienceId}", experience.ExperienceId);
                throw;
            }
        }

        /// <summary>
        /// Load all experiences from a date range
        /// Used by Lab for training
        /// </summary>
        public async Task<List<TradingExperience>> LoadExperiencesAsync(DateTime from, DateTime to)
        {
            var experiences = new List<TradingExperience>();

            try
            {
                // Get all JSON files in directory
                var files = Directory.GetFiles(_experienceDirectory, "*.json", SearchOption.TopDirectoryOnly);

                foreach (var file in files)
                {
                    try
                    {
                        // Parse filename to get timestamp
                        var filename = Path.GetFileName(file);
                        var timestampStr = filename.Split('_')[0]; // yyyy-MM-dd
                        
                        if (!DateTime.TryParse(timestampStr, out var fileDate))
                        {
                            continue; // Skip invalid filenames
                        }

                        // Filter by date range
                        if (fileDate < from.Date || fileDate > to.Date)
                        {
                            continue;
                        }

                        // Read and deserialize
                        var json = await File.ReadAllTextAsync(file).ConfigureAwait(false);
                        var experience = JsonSerializer.Deserialize<TradingExperience>(json, _jsonOptions);

                        if (experience != null)
                        {
                            experiences.Add(experience);
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "⚠️ [EXPERIENCE-REPO] Failed to load file: {File}", file);
                        // Continue with next file
                    }
                }

                _logger.LogInformation(
                    "📖 [EXPERIENCE-REPO] Loaded {Count} experiences from {From:yyyy-MM-dd} to {To:yyyy-MM-dd}",
                    experiences.Count,
                    from,
                    to
                );

                return experiences.OrderBy(e => e.Timestamp).ToList();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [EXPERIENCE-REPO] Failed to load experiences");
                throw;
            }
        }

        /// <summary>
        /// Load experiences from last N days
        /// Convenience method for Lab training
        /// </summary>
        public Task<List<TradingExperience>> LoadRecentExperiencesAsync(int days)
        {
            var to = DateTime.UtcNow;
            var from = to.AddDays(-days);
            return LoadExperiencesAsync(from, to);
        }

        /// <summary>
        /// Get count of experiences in a date range
        /// </summary>
        public int GetExperienceCount(DateTime from, DateTime to)
        {
            try
            {
                var files = Directory.GetFiles(_experienceDirectory, "*.json", SearchOption.TopDirectoryOnly);
                
                int count = 0;
                foreach (var file in files)
                {
                    var filename = Path.GetFileName(file);
                    var timestampStr = filename.Split('_')[0];
                    
                    if (DateTime.TryParse(timestampStr, out var fileDate))
                    {
                        if (fileDate >= from.Date && fileDate <= to.Date)
                        {
                            count++;
                        }
                    }
                }

                return count;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [EXPERIENCE-REPO] Failed to count experiences");
                return 0;
            }
        }

        /// <summary>
        /// Clean up old experiences (older than retention days)
        /// </summary>
        public async Task CleanupOldExperiencesAsync(int retentionDays)
        {
            try
            {
                var cutoffDate = DateTime.UtcNow.AddDays(-retentionDays);
                var files = Directory.GetFiles(_experienceDirectory, "*.json", SearchOption.TopDirectoryOnly);
                
                int deletedCount = 0;
                foreach (var file in files)
                {
                    var filename = Path.GetFileName(file);
                    var timestampStr = filename.Split('_')[0];
                    
                    if (DateTime.TryParse(timestampStr, out var fileDate))
                    {
                        if (fileDate < cutoffDate.Date)
                        {
                            File.Delete(file);
                            deletedCount++;
                        }
                    }
                }

                if (deletedCount > 0)
                {
                    _logger.LogInformation(
                        "🗑️ [EXPERIENCE-REPO] Cleaned up {Count} old experiences (older than {Days} days)",
                        deletedCount,
                        retentionDays
                    );
                }

                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [EXPERIENCE-REPO] Failed to cleanup old experiences");
            }
        }
    }
}
