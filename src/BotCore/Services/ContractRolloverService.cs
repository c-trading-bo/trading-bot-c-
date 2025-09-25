using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using BotCore.Configuration;

namespace BotCore.Services
{
    /// <summary>
    /// Contract rollover service for automatic front month detection and management
    /// Handles ES, NQ contract rollovers to ensure trading on active contracts
    /// </summary>
    public interface IContractRolloverService
    {
        Task<string> GetCurrentFrontMonthContractAsync(string baseSymbol);
        Task<ContractInfo> GetContractInfoAsync(string contractSymbol);
        Task<bool> ShouldRolloverAsync(string currentContract);
        Task<string> GetNextContractAsync(string currentContract);
        Task<List<ContractInfo>> GetActiveContractsAsync(string baseSymbol);
        DateTime GetContractExpirationDate(string contractSymbol);
        Task MonitorRolloverRequirementsAsync(CancellationToken cancellationToken);
    }

    /// <summary>
    /// Comprehensive contract rollover service implementation
    /// </summary>
    public class ContractRolloverService : IContractRolloverService
    {
        private readonly ILogger<ContractRolloverService> _logger;
        private readonly DataFlowEnhancementConfiguration _config;
        private readonly Dictionary<string, ContractSpec> _contractSpecs;

        public ContractRolloverService(
            ILogger<ContractRolloverService> logger,
            IOptions<DataFlowEnhancementConfiguration> config)
        {
            _logger = logger;
            _config = config.Value;
            _contractSpecs = InitializeContractSpecs();
        }

        /// <summary>
        /// Get current front month contract for a base symbol
        /// </summary>
        public Task<string> GetCurrentFrontMonthContractAsync(string baseSymbol)
        {
            try
            {
                _logger.LogDebug("[CONTRACT-ROLLOVER] Getting front month contract for {BaseSymbol}", baseSymbol);

                // Check configured mapping first
                if (_config.FrontMonthMapping.TryGetValue(baseSymbol.ToUpper(), out var configuredContract))
                {
                    // Verify the configured contract is still valid
                    if (IsContractActiveAsync(configuredContract).Result)
                    {
                        return Task.FromResult(configuredContract);
                    }
                    else
                    {
                        _logger.LogWarning("[CONTRACT-ROLLOVER] Configured contract {Contract} is no longer active, calculating new front month", configuredContract);
                    }
                }

                // Calculate front month based on current date
                var frontMonth = CalculateFrontMonthContract(baseSymbol);
                
                _logger.LogInformation("[CONTRACT-ROLLOVER] Front month contract for {BaseSymbol}: {FrontMonth}", baseSymbol, frontMonth);
                
                return Task.FromResult(frontMonth);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[CONTRACT-ROLLOVER] Error getting front month contract for {BaseSymbol}", baseSymbol);
                
                // Return fallback
                var fallback = _config.FrontMonthMapping.TryGetValue(baseSymbol.ToUpper(), out var fb) 
                    ? fb 
                    : $"{baseSymbol}Z25"; // Default to December 2025
                return Task.FromResult(fallback);
            }
        }

        /// <summary>
        /// Get detailed contract information
        /// </summary>
        public async Task<ContractInfo> GetContractInfoAsync(string contractSymbol)
        {
            try
            {
                var baseSymbol = ExtractBaseSymbol(contractSymbol);
                var monthCode = ExtractMonthCode(contractSymbol);
                var year = ExtractYear(contractSymbol);

                if (!_contractSpecs.TryGetValue(baseSymbol, out var spec))
                {
                    throw new ArgumentException($"Unknown contract base symbol: {baseSymbol}");
                }

                var expirationDate = CalculateExpirationDate(monthCode, year);
                var isActive = await IsContractActiveAsync(contractSymbol).ConfigureAwait(false);
                var daysToExpiration = (expirationDate - DateTime.UtcNow).Days;

                return new ContractInfo
                {
                    ContractSymbol = contractSymbol,
                    BaseSymbol = baseSymbol,
                    MonthCode = monthCode,
                    Year = year,
                    ExpirationDate = expirationDate,
                    DaysToExpiration = daysToExpiration,
                    IsActive = isActive,
                    IsFrontMonth = daysToExpiration > 0 && daysToExpiration <= 60, // Simplified logic
                    TickSize = spec.TickSize,
                    ContractSize = spec.ContractSize,
                    Currency = spec.Currency
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[CONTRACT-INFO] Error getting contract info for {ContractSymbol}", contractSymbol);
                throw;
            }
        }

        /// <summary>
        /// Determine if a contract should be rolled over
        /// </summary>
        public Task<bool> ShouldRolloverAsync(string currentContract)
        {
            try
            {
                if (!_config.EnableContractRollover)
                    return Task.FromResult(false);

                var contractInfo = GetContractInfoAsync(currentContract).Result;
                var shouldRollover = contractInfo.DaysToExpiration <= _config.ContractRolloverDays;

                if (shouldRollover)
                {
                    _logger.LogInformation("[CONTRACT-ROLLOVER] Contract {Contract} should be rolled over ({Days} days to expiration)",
                        currentContract, contractInfo.DaysToExpiration);
                }

                return Task.FromResult(shouldRollover);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[CONTRACT-ROLLOVER] Error checking rollover for {Contract}", currentContract);
                return Task.FromResult(false);
            }
        }

        /// <summary>
        /// Get the next contract after the current one
        /// </summary>
        public Task<string> GetNextContractAsync(string currentContract)
        {
            try
            {
                var baseSymbol = ExtractBaseSymbol(currentContract);
                var currentMonthCode = ExtractMonthCode(currentContract);
                var currentYear = ExtractYear(currentContract);

                // Get the contract spec for month sequence
                if (!_contractSpecs.TryGetValue(baseSymbol, out var spec))
                {
                    throw new ArgumentException($"Unknown contract base symbol: {baseSymbol}");
                }

                // Find current month in the sequence
                var currentIndex = Array.IndexOf(spec.MonthSequence, currentMonthCode);
                if (currentIndex == -1)
                {
                    throw new ArgumentException($"Invalid month code {currentMonthCode} for {baseSymbol}");
                }

                // Get next month
                string nextMonthCode;
                int nextYear;

                if (currentIndex < spec.MonthSequence.Length - 1)
                {
                    // Next month in same year
                    nextMonthCode = spec.MonthSequence[currentIndex + 1];
                    nextYear = currentYear;
                }
                else
                {
                    // First month of next year
                    nextMonthCode = spec.MonthSequence[0];
                    nextYear = currentYear + 1;
                }

                var nextContract = $"{baseSymbol}{nextMonthCode}{nextYear % 100:D2}";
                
                _logger.LogInformation("[CONTRACT-ROLLOVER] Next contract after {Current}: {Next}", currentContract, nextContract);
                
                return Task.FromResult(nextContract);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[CONTRACT-ROLLOVER] Error getting next contract for {Current}", currentContract);
                throw;
            }
        }

        /// <summary>
        /// Get list of active contracts for a base symbol
        /// </summary>
        public async Task<List<ContractInfo>> GetActiveContractsAsync(string baseSymbol)
        {
            try
            {
                var activeContracts = new List<ContractInfo>();

                if (!_contractSpecs.TryGetValue(baseSymbol.ToUpper(), out var spec))
                {
                    _logger.LogWarning("[CONTRACT-LIST] Unknown base symbol: {BaseSymbol}", baseSymbol);
                    return activeContracts;
                }

                var currentDate = DateTime.UtcNow;
                var currentYear = currentDate.Year;

                // Check contracts for current and next year
                for (int yearOffset = 0; yearOffset <= 1; yearOffset++)
                {
                    var year = currentYear + yearOffset;
                    
                    foreach (var monthCode in spec.MonthSequence)
                    {
                        var contractSymbol = $"{baseSymbol}{monthCode}{year % 100:D2}";
                        var expirationDate = CalculateExpirationDate(monthCode, year);
                        
                        // Only include contracts that haven't expired and are within 12 months
                        if (expirationDate > currentDate && expirationDate <= currentDate.AddMonths(12))
                        {
                            var contractInfo = await GetContractInfoAsync(contractSymbol).ConfigureAwait(false);
                            activeContracts.Add(contractInfo);
                        }
                    }
                }

                return activeContracts.OrderBy(c => c.ExpirationDate).ToList();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[CONTRACT-LIST] Error getting active contracts for {BaseSymbol}", baseSymbol);
                return new List<ContractInfo>();
            }
        }

        /// <summary>
        /// Get contract expiration date
        /// </summary>
        public DateTime GetContractExpirationDate(string contractSymbol)
        {
            var baseSymbol = ExtractBaseSymbol(contractSymbol);
            var monthCode = ExtractMonthCode(contractSymbol);
            var year = ExtractYear(contractSymbol);

            return CalculateExpirationDate(monthCode, year);
        }

        /// <summary>
        /// Monitor rollover requirements continuously
        /// </summary>
        public async Task MonitorRolloverRequirementsAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("[CONTRACT-MONITOR] Starting contract rollover monitoring");

            try
            {
                while (!cancellationToken.IsCancellationRequested)
                {
                    await CheckRolloverRequirementsAsync().ConfigureAwait(false);
                    await Task.Delay(TimeSpan.FromHours(4), cancellationToken).ConfigureAwait(false); // Check every 4 hours
                }
            }
            catch (OperationCanceledException)
            {
                _logger.LogInformation("[CONTRACT-MONITOR] Contract rollover monitoring stopped");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[CONTRACT-MONITOR] Error in contract rollover monitoring");
            }
        }

        #region Private Methods

        /// <summary>
        /// Initialize contract specifications
        /// </summary>
        private Dictionary<string, ContractSpec> InitializeContractSpecs()
        {
            return new Dictionary<string, ContractSpec>
            {
                ["ES"] = new ContractSpec
                {
                    BaseSymbol = "ES",
                    FullName = "E-mini S&P 500",
                    MonthSequence = new[] { "H", "M", "U", "Z" }, // Mar, Jun, Sep, Dec
                    TickSize = 0.25m,
                    ContractSize = 50,
                    Currency = "USD",
                    ExpirationRule = ContractExpirationRule.ThirdFridayOfMonth
                },
                ["NQ"] = new ContractSpec
                {
                    BaseSymbol = "NQ",
                    FullName = "E-mini NASDAQ-100",
                    MonthSequence = new[] { "H", "M", "U", "Z" }, // Mar, Jun, Sep, Dec
                    TickSize = 0.25m,
                    ContractSize = 20,
                    Currency = "USD",
                    ExpirationRule = ContractExpirationRule.ThirdFridayOfMonth
                }
            };
        }

        /// <summary>
        /// Calculate front month contract based on current date
        /// </summary>
        private string CalculateFrontMonthContract(string baseSymbol)
        {
            if (!_contractSpecs.TryGetValue(baseSymbol.ToUpper(), out var spec))
            {
                throw new ArgumentException($"Unknown base symbol: {baseSymbol}");
            }

            var currentDate = DateTime.UtcNow;
            var currentMonth = currentDate.Month;
            
            // Map months to contract months for quarterly contracts
            var quarterlyMonths = new[] { 3, 6, 9, 12 }; // Mar, Jun, Sep, Dec
            var currentQuarterIndex = quarterlyMonths.Where(m => m >= currentMonth).FirstOrDefault();
            
            if (currentQuarterIndex == 0) // Past December, go to next year March
            {
                return $"{baseSymbol}H{(currentDate.Year + 1) % 100:D2}";
            }

            var monthIndex = Array.IndexOf(quarterlyMonths, currentQuarterIndex);
            var monthCode = spec.MonthSequence[monthIndex];
            
            // Check if we're too close to expiration and should use next contract
            var expirationDate = CalculateExpirationDate(monthCode, currentDate.Year);
            if ((expirationDate - currentDate).Days <= _config.ContractRolloverDays)
            {
                // Move to next quarter
                monthIndex = (monthIndex + 1) % spec.MonthSequence.Length;
                monthCode = spec.MonthSequence[monthIndex];
                var year = monthIndex == 0 ? currentDate.Year + 1 : currentDate.Year;
                return $"{baseSymbol}{monthCode}{year % 100:D2}";
            }

            return $"{baseSymbol}{monthCode}{currentDate.Year % 100:D2}";
        }

        /// <summary>
        /// Calculate expiration date for a contract
        /// </summary>
        private DateTime CalculateExpirationDate(string monthCode, int year)
        {
            var month = MonthCodeToMonth(monthCode);
            
            // For ES/NQ: Third Friday of the month
            var thirdFriday = GetThirdFridayOfMonth(year, month);
            
            // Set expiration time to 9:30 AM ET (market open)
            return new DateTime(thirdFriday.Year, thirdFriday.Month, thirdFriday.Day, 9, 30, 0, DateTimeKind.Unspecified);
        }

        /// <summary>
        /// Get third Friday of a month
        /// </summary>
        private DateTime GetThirdFridayOfMonth(int year, int month)
        {
            var firstDay = new DateTime(year, month, 1);
            var firstFriday = firstDay.AddDays((DayOfWeek.Friday - firstDay.DayOfWeek + 7) % 7);
            return firstFriday.AddDays(14); // Third Friday is 2 weeks after first Friday
        }

        /// <summary>
        /// Convert month code to month number
        /// </summary>
        private int MonthCodeToMonth(string monthCode)
        {
            return monthCode.ToUpper() switch
            {
                "F" => 1,  // January
                "G" => 2,  // February
                "H" => 3,  // March
                "J" => 4,  // April
                "K" => 5,  // May
                "M" => 6,  // June
                "N" => 7,  // July
                "Q" => 8,  // August
                "U" => 9,  // September
                "V" => 10, // October
                "X" => 11, // November
                "Z" => 12, // December
                _ => throw new ArgumentException($"Invalid month code: {monthCode}")
            };
        }

        /// <summary>
        /// Extract base symbol from contract symbol (e.g., "ESZ3" -> "ES")
        /// </summary>
        private string ExtractBaseSymbol(string contractSymbol)
        {
            if (contractSymbol.Length < 2)
                throw new ArgumentException("Invalid contract symbol format");

            // Handle ES/NQ (2 chars)
            return contractSymbol[..2];
        }

        /// <summary>
        /// Extract month code from contract symbol
        /// </summary>
        private string ExtractMonthCode(string contractSymbol)
        {
            var baseLength = ExtractBaseSymbol(contractSymbol).Length;
            if (contractSymbol.Length <= baseLength)
                throw new ArgumentException("Invalid contract symbol format");

            return contractSymbol[baseLength].ToString();
        }

        /// <summary>
        /// Extract year from contract symbol
        /// </summary>
        private int ExtractYear(string contractSymbol)
        {
            var baseLength = ExtractBaseSymbol(contractSymbol).Length;
            if (contractSymbol.Length < baseLength + 3)
                throw new ArgumentException("Invalid contract symbol format");

            var yearStr = contractSymbol.Substring(baseLength + 1, 2);
            if (!int.TryParse(yearStr, out var year))
                throw new ArgumentException("Invalid year format in contract symbol");

            // Convert 2-digit year to 4-digit year
            var currentYear = DateTime.UtcNow.Year;
            var currentCentury = (currentYear / 100) * 100;
            var fullYear = currentCentury + year;

            // If the year is more than 50 years in the past, assume next century
            if (fullYear < currentYear - 50)
                fullYear += 100;

            return fullYear;
        }

        /// <summary>
        /// Check if a contract is still active (not expired)
        /// </summary>
        private Task<bool> IsContractActiveAsync(string contractSymbol)
        {
            try
            {
                var expirationDate = GetContractExpirationDate(contractSymbol);
                return Task.FromResult(expirationDate > DateTime.UtcNow);
            }
            catch
            {
                return Task.FromResult(false);
            }
        }

        /// <summary>
        /// Check rollover requirements for all configured contracts
        /// </summary>
        private async Task CheckRolloverRequirementsAsync()
        {
            try
            {
                var baseSymbols = new[] { "ES", "NQ" };

                foreach (var baseSymbol in baseSymbols)
                {
                    var frontMonth = await GetCurrentFrontMonthContractAsync(baseSymbol).ConfigureAwait(false);
                    var shouldRollover = await ShouldRolloverAsync(frontMonth).ConfigureAwait(false);

                    if (shouldRollover)
                    {
                        var nextContract = await GetNextContractAsync(frontMonth).ConfigureAwait(false);
                        _logger.LogWarning("[CONTRACT-MONITOR] ⚠️ Rollover required: {BaseSymbol} from {Current} to {Next}",
                            baseSymbol, frontMonth, nextContract);

                        // Update configuration mapping
                        _config.FrontMonthMapping[baseSymbol] = nextContract;
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[CONTRACT-MONITOR] Error checking rollover requirements");
            }
        }

        #endregion
    }

    #region Supporting Models

    /// <summary>
    /// Contract specification
    /// </summary>
    public class ContractSpec
    {
        public string BaseSymbol { get; set; } = string.Empty;
        public string FullName { get; set; } = string.Empty;
        public string[] MonthSequence { get; set; } = Array.Empty<string>();
        public decimal TickSize { get; set; }
        public int ContractSize { get; set; }
        public string Currency { get; set; } = "USD";
        public ContractExpirationRule ExpirationRule { get; set; }
    }

    /// <summary>
    /// Contract information
    /// </summary>
    public class ContractInfo
    {
        public string ContractSymbol { get; set; } = string.Empty;
        public string BaseSymbol { get; set; } = string.Empty;
        public string MonthCode { get; set; } = string.Empty;
        public int Year { get; set; }
        public DateTime ExpirationDate { get; set; }
        public int DaysToExpiration { get; set; }
        public bool IsActive { get; set; }
        public bool IsFrontMonth { get; set; }
        public decimal TickSize { get; set; }
        public int ContractSize { get; set; }
        public string Currency { get; set; } = string.Empty;
    }

    /// <summary>
    /// Contract expiration rules
    /// </summary>
    public enum ContractExpirationRule
    {
        ThirdFridayOfMonth,
        LastTradingDayOfMonth,
        CustomRule
    }

    #endregion
}