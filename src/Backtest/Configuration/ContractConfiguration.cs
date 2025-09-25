using Microsoft.Extensions.Configuration;

namespace TradingBot.Backtest.Configuration
{
    /// <summary>
    /// Configuration for contract-specific trading parameters
    /// Replaces hardcoded trading values with config-driven parameters
    /// Addresses production requirement: "No hardcoded trading values"
    /// </summary>
    public class ContractConfiguration
    {
        /// <summary>
        /// Contract symbol (ES, NQ)
        /// </summary>
        public string Symbol { get; set; } = string.Empty;

        /// <summary>
        /// Tick size in points for spread calculation
        /// </summary>
        public decimal TickSize { get; set; }

        /// <summary>
        /// Point value in USD (how much 1 point is worth)
        /// </summary>
        public decimal PointValue { get; set; }

        /// <summary>
        /// Minimum spread multiplier (e.g., 1.0 for 1 tick spread minimum)
        /// </summary>
        public decimal MinSpreadMultiplier { get; set; } = 1.0m;

        /// <summary>
        /// Description of the contract
        /// </summary>
        public string Description { get; set; } = string.Empty;
    }

    /// <summary>
    /// Configuration for all supported contracts (ES and NQ only)
    /// </summary>
    public class ContractConfigurationOptions
    {
        public const string SectionName = "ContractConfiguration";

        /// <summary>
        /// Contract configurations keyed by symbol
        /// </summary>
        public Dictionary<string, ContractConfiguration> Contracts { get; set; } = new();

        /// <summary>
        /// Get configuration for a specific contract
        /// </summary>
        public ContractConfiguration GetContract(string symbol)
        {
            if (Contracts.TryGetValue(symbol, out var config))
            {
                return config;
            }

            throw new ArgumentException($"No configuration found for contract {symbol}. Only ES and NQ contracts are supported.", nameof(symbol));
        }

        /// <summary>
        /// Validate that only ES and NQ contracts are configured
        /// </summary>
        public void ValidateProductionConfiguration()
        {
            var supportedContracts = new[] { "ES", "NQ" };
            var configuredContracts = Contracts.Keys.ToArray();

            // Check for unsupported contracts
            var unsupportedContracts = configuredContracts.Except(supportedContracts).ToArray();
            if (unsupportedContracts.Any())
            {
                throw new InvalidOperationException($"Unsupported contracts in configuration: {string.Join(", ", unsupportedContracts)}. Only ES and NQ are supported.");
            }

            // Check that required contracts are present
            foreach (var requiredContract in supportedContracts)
            {
                if (!Contracts.ContainsKey(requiredContract))
                {
                    throw new InvalidOperationException($"Missing required contract configuration: {requiredContract}");
                }
            }

            // Validate individual contract configurations
            foreach (var contract in Contracts.Values)
            {
                if (contract.TickSize <= 0)
                {
                    throw new InvalidOperationException($"Invalid tick size for contract {contract.Symbol}: {contract.TickSize}. Must be > 0.");
                }

                if (contract.PointValue <= 0)
                {
                    throw new InvalidOperationException($"Invalid point value for contract {contract.Symbol}: {contract.PointValue}. Must be > 0.");
                }

                if (contract.MinSpreadMultiplier <= 0)
                {
                    throw new InvalidOperationException($"Invalid min spread multiplier for contract {contract.Symbol}: {contract.MinSpreadMultiplier}. Must be > 0.");
                }
            }
        }
    }
}