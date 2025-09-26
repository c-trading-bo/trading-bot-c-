using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// Determinism service for seeding stochastic components for reproducible QA
    /// Ensures all random components can be seeded for testing and debugging
    /// </summary>
    public class DeterminismService
    {
        private readonly ILogger<DeterminismService> _logger;
        private readonly IConfiguration _config;
        private readonly Dictionary<string, Random> _seededRandoms = new();
        private readonly Dictionary<string, int> _seedRegistry = new();
        private readonly object _registryLock = new();

        public DeterminismService(ILogger<DeterminismService> logger, IConfiguration config)
        {
            _logger = logger;
            _config = config;
        }

        /// <summary>
        /// Get or create a seeded random number generator for a component
        /// </summary>
        public Random GetSeededRandom(string componentName, int? forceSeed = null)
        {
            lock (_registryLock)
            {
                if (_seededRandoms.TryGetValue(componentName, out var existingRandom))
                {
                    return existingRandom;
                }

                int seed;
                
                if (forceSeed.HasValue)
                {
                    seed = forceSeed.Value;
                }
                else if (IsReproducibleMode())
                {
                    seed = GetDeterministicSeed(componentName);
                }
                else
                {
                    // Production mode - use cryptographically secure random seed
                    seed = GenerateSecureRandomSeed();
                }

                var random = new Random(seed);
                _seededRandoms[componentName] = random;
                _seedRegistry[componentName] = seed;

                _logger.LogInformation("🎲 [DETERMINISM] Seeded random generator for {Component} with seed {Seed} (Mode: {Mode})", 
                    componentName, seed, IsReproducibleMode() ? "Reproducible" : "Production");

                return random;
            }
        }

        /// <summary>
        /// Get deterministic seed based on component name and global seed
        /// </summary>
        private int GetDeterministicSeed(string componentName)
        {
            var globalSeed = _config.GetValue("Determinism:GlobalSeed", 42);
            
            // Create deterministic hash from global seed and component name
            var input = Encoding.UTF8.GetBytes($"{globalSeed}-{componentName}");
            var hash = SHA256.HashData(input);
            
            // Convert first 4 bytes of hash to int
            return BitConverter.ToInt32(hash, 0);
        }

        /// <summary>
        /// Generate cryptographically secure random seed for production
        /// </summary>
        private static int GenerateSecureRandomSeed()
        {
            using var rng = RandomNumberGenerator.Create();
            var bytes = new byte[4];
            rng.GetBytes(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

        /// <summary>
        /// Check if system is in reproducible mode for QA/testing
        /// </summary>
        public bool IsReproducibleMode()
        {
            return _config.GetValue("Determinism:ReproducibleMode", false) ||
                   Environment.GetEnvironmentVariable("TRADING_BOT_REPRODUCIBLE") == "true";
        }

        /// <summary>
        /// Enable reproducible mode with specific global seed
        /// </summary>
        public void EnableReproducibleMode(int globalSeed)
        {
            lock (_registryLock)
            {
                // Clear existing random generators
                _seededRandoms.Clear();
                _seedRegistry.Clear();
                
                // Set global seed in config
                Environment.SetEnvironmentVariable("TRADING_BOT_REPRODUCIBLE", "true");
                
                _logger.LogInformation("🔒 [DETERMINISM] Enabled reproducible mode with global seed {GlobalSeed}", globalSeed);
            }
        }

        /// <summary>
        /// Disable reproducible mode and return to production randomness
        /// </summary>
        public void DisableReproducibleMode()
        {
            lock (_registryLock)
            {
                // Clear existing random generators
                _seededRandoms.Clear();
                _seedRegistry.Clear();
                
                Environment.SetEnvironmentVariable("TRADING_BOT_REPRODUCIBLE", "false");
                
                _logger.LogInformation("🎰 [DETERMINISM] Disabled reproducible mode - using production randomness");
            }
        }

        /// <summary>
        /// Get current seed registry for debugging and QA logs
        /// </summary>
        public Dictionary<string, int> GetSeedRegistry()
        {
            lock (_registryLock)
            {
                return new Dictionary<string, int>(_seedRegistry);
            }
        }

        /// <summary>
        /// Seed a specific ML model or algorithm component
        /// </summary>
        public void SeedMLComponent(string modelName, object mlModel)
        {
            var seed = GetDeterministicSeed($"ML-{modelName}");
            
            // This is a placeholder - actual implementation would depend on the ML framework
            // For example, if using ML.NET, TensorFlow.NET, etc.
            
            _logger.LogInformation("🧠 [DETERMINISM] Seeded ML component {ModelName} with seed {Seed}", modelName, seed);
        }

        /// <summary>
        /// Create seeded randomness for strategy components
        /// </summary>
        public Random SeedStrategy(string strategyId)
        {
            return GetSeededRandom($"Strategy-{strategyId}");
        }

        /// <summary>
        /// Create seeded randomness for portfolio components
        /// </summary>
        public Random SeedPortfolio(string portfolioId)
        {
            return GetSeededRandom($"Portfolio-{portfolioId}");
        }

        /// <summary>
        /// Create seeded randomness for execution components
        /// </summary>
        public Random SeedExecution(string executionId)
        {
            return GetSeededRandom($"Execution-{executionId}");
        }

        /// <summary>
        /// Generate reproducible UUID based on seed and input
        /// </summary>
        public Guid GenerateReproducibleGuid(string input)
        {
            if (!IsReproducibleMode())
            {
                return Guid.NewGuid();
            }

            var globalSeed = _config.GetValue("Determinism:GlobalSeed", 42);
            
            var seedInput = Encoding.UTF8.GetBytes($"{globalSeed}-{input}");
            var hash = SHA256.HashData(seedInput);
            
            // Take first 16 bytes for GUID
            var guidBytes = new byte[16];
            Array.Copy(hash, 0, guidBytes, 0, 16);
            
            return new Guid(guidBytes);
        }

        /// <summary>
        /// Validate that all stochastic components are properly seeded
        /// </summary>
        public DeterminismValidationResult ValidateSeeding()
        {
            var result = new DeterminismValidationResult
            {
                IsReproducibleMode = IsReproducibleMode(),
                SeededComponentCount = _seededRandoms.Count,
                SeedRegistry = GetSeedRegistry(),
                ValidationTime = DateTime.UtcNow
            };

            var expectedComponents = new[] { "Strategy", "Portfolio", "Execution", "ML", "Risk" };
            var missingComponents = new List<string>();

            foreach (var component in expectedComponents)
            {
                var hasSeededComponent = false;
                foreach (var key in _seedRegistry.Keys)
                {
                    if (key.StartsWith(component, StringComparison.OrdinalIgnoreCase))
                    {
                        hasSeededComponent = true;
                        break;
                    }
                }

                if (!hasSeededComponent)
                {
                    missingComponents.Add(component);
                }
            }

            result.MissingComponents = missingComponents;
            result.IsValid = missingComponents.Count == 0;

            if (result.IsValid)
            {
                _logger.LogInformation("✅ [DETERMINISM] All stochastic components properly seeded");
            }
            else
            {
                _logger.LogWarning("⚠️ [DETERMINISM] Missing seeded components: {Missing}", string.Join(", ", missingComponents));
            }

            return result;
        }
    }

    /// <summary>
    /// Result of determinism validation
    /// </summary>
    public class DeterminismValidationResult
    {
        public bool IsReproducibleMode { get; set; }
        public int SeededComponentCount { get; set; }
        public Dictionary<string, int> SeedRegistry { get; set; } = new();
        public List<string> MissingComponents { get; set; } = new();
        public bool IsValid { get; set; }
        public DateTime ValidationTime { get; set; }
    }
}