using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// Secrets and endpoint validation service
    /// Ensures all sensitive data comes from environment/secret store only
    /// Validates configuration at startup and prevents secret exposure
    /// </summary>
    public class SecretsValidationService
    {
        // API Key Validation Length Constants
        private const int TopstepXApiKeyMinLength = 32;
        private const int TopstepXApiSecretMinLength = 32;  
        private const int DefaultApiKeyMinLength = 16;
        
        // Secret Masking Constants
        private const int SecretPrefixLength = 3;           // Characters to show at start of masked secret
        private const int SecretSuffixLength = 3;           // Characters to show at end of masked secret 
        private const int SecretMaskingOverhead = 6;        // Total prefix + suffix characters (3 + 3)
        
        private readonly ILogger<SecretsValidationService> _logger;
        private readonly IConfiguration _config;
        private readonly List<SecretValidationRule> _validationRules;

        public SecretsValidationService(ILogger<SecretsValidationService> logger, IConfiguration config)
        {
            _logger = logger;
            _config = config;
            _validationRules = InitializeValidationRules();
        }

        /// <summary>
        /// Validate all secrets and endpoints at startup
        /// </summary>
        public StartupValidationResult ValidateAtStartup()
        {
            var result = new StartupValidationResult
            {
                ValidationTime = DateTime.UtcNow
            };

            _logger.LogInformation("🔐 [SECRETS] Starting secrets and endpoint validation...");

            try
            {
                // Validate API keys and tokens
                ValidateApiCredentials(result);
                
                // Validate endpoint configurations
                ValidateEndpointConfiguration(result);
                
                // Validate database connection strings
                ValidateDatabaseConnections(result);
                
                // Check for hardcoded secrets in configuration
                DetectHardcodedSecrets(result);
                
                // Validate secret store connectivity
                ValidateSecretStoreAccess(result);

                result.IsValid = result.Errors.Count == 0 && result.MissingSecrets.Count == 0;

                if (result.IsValid)
                {
                    _logger.LogInformation("✅ [SECRETS] All secrets and endpoints validated successfully");
                }
                else
                {
                    _logger.LogError("🚨 [SECRETS] Validation failed: {ErrorCount} errors, {MissingCount} missing secrets",
                        result.Errors.Count, result.MissingSecrets.Count);
                }
            }
            catch (InvalidOperationException ex)
            {
                _logger.LogError(ex, "🚨 [SECRETS] Invalid operation during secrets validation");
                result.AddError($"Critical validation error: {ex.Message}");
                result.IsValid = false;
            }
            catch (UnauthorizedAccessException ex)
            {
                _logger.LogError(ex, "🚨 [SECRETS] Access denied during secrets validation");
                result.AddError($"Critical validation error: {ex.Message}");
                result.IsValid = false;
            }
            catch (ArgumentException ex)
            {
                _logger.LogError(ex, "🚨 [SECRETS] Invalid argument during secrets validation");
                result.AddError($"Critical validation error: {ex.Message}");
                result.IsValid = false;
            }

            return result;
        }

        /// <summary>
        /// Validate API credentials from environment variables
        /// </summary>
        private void ValidateApiCredentials(StartupValidationResult result)
        {
            var requiredApiKeys = new[]
            {
                "TOPSTEPX_API_KEY",
                "TOPSTEPX_API_SECRET",
                "DATA_FEED_API_KEY",
                "ML_SERVICE_API_KEY"
            };

            foreach (var keyName in requiredApiKeys)
            {
                var keyValue = Environment.GetEnvironmentVariable(keyName);
                
                if (string.IsNullOrEmpty(keyValue))
                {
                    result.AddMissingSecret(keyName);
                    _logger.LogError("🚨 [SECRETS] Missing required API key: {KeyName}", keyName);
                }
                else
                {
                    // Validate key format
                    if (IsValidApiKey(keyName, keyValue))
                    {
                        result.AddValidatedSecret(keyName);
                        _logger.LogDebug("✅ [SECRETS] Valid API key: {KeyName}", keyName);
                    }
                    else
                    {
                        result.AddInvalidSecret(keyName);
                        _logger.LogError("🚨 [SECRETS] Invalid API key format: {KeyName}", keyName);
                    }
                }
            }
        }

        /// <summary>
        /// Validate endpoint configurations are not hardcoded
        /// </summary>
        private void ValidateEndpointConfiguration(StartupValidationResult result)
        {
            var endpointKeys = new[]
            {
                "Endpoints:TopstepXApiBaseUrl",
                "Endpoints:DataFeedEndpoint",
                "Endpoints:MLServiceEndpoint"
            };

            foreach (var endpointKey in endpointKeys)
            {
                var endpointValue = _config.GetValue<string>(endpointKey);
                
                if (string.IsNullOrEmpty(endpointValue))
                {
                    result.AddMissingSecret(endpointKey);
                    _logger.LogError("🚨 [SECRETS] Missing endpoint configuration: {Key}", endpointKey);
                }
                else if (IsHardcodedEndpoint(endpointValue))
                {
                    result.AddError($"Hardcoded endpoint detected: {endpointKey}");
                    _logger.LogError("🚨 [SECRETS] Hardcoded endpoint detected: {Key} = {Value}", endpointKey, MaskSensitiveValue(endpointValue));
                }
                else
                {
                    result.AddValidatedSecret(endpointKey);
                    _logger.LogDebug("✅ [SECRETS] Valid endpoint configuration: {Key}", endpointKey);
                }
            }
        }

        /// <summary>
        /// Validate database connection strings
        /// </summary>
        private void ValidateDatabaseConnections(StartupValidationResult result)
        {
            var connectionStrings = new[]
            {
                "ConnectionStrings:TradingDatabase",
                "ConnectionStrings:LoggingDatabase"
            };

            foreach (var connKey in connectionStrings)
            {
                var connString = _config.GetConnectionString(connKey.Split(':')[1]);
                
                if (string.IsNullOrEmpty(connString))
                {
                    // Connection strings might be optional
                    _logger.LogWarning("⚠️ [SECRETS] Optional connection string not configured: {Key}", connKey);
                }
                else if (ContainsHardcodedCredentials(connString))
                {
                    result.AddError($"Hardcoded credentials in connection string: {connKey}");
                    _logger.LogError("🚨 [SECRETS] Hardcoded credentials in connection string: {Key}", connKey);
                }
                else
                {
                    result.AddValidatedSecret(connKey);
                    _logger.LogDebug("✅ [SECRETS] Valid connection string: {Key}", connKey);
                }
            }
        }

        /// <summary>
        /// Detect hardcoded secrets in configuration values
        /// </summary>
        private void DetectHardcodedSecrets(StartupValidationResult result)
        {
            var configRoot = _config as IConfigurationRoot;
            if (configRoot == null) return;

            foreach (var provider in configRoot.Providers)
            {
                // Check for patterns that indicate hardcoded secrets
                foreach (var rule in _validationRules)
                {
                    if (rule.Pattern.IsMatch(provider.ToString() ?? ""))
                    {
                        result.AddError($"Potential hardcoded secret detected: {rule.Name}");
                        _logger.LogError("🚨 [SECRETS] Potential hardcoded secret detected: {Rule}", rule.Name);
                    }
                }
            }
        }

        /// <summary>
        /// Validate secret store accessibility
        /// </summary>
        private void ValidateSecretStoreAccess(StartupValidationResult result)
        {
            try
            {
                // Test access to secret store (Azure Key Vault, HashiCorp Vault, etc.)
                // This is a placeholder - actual implementation depends on secret store provider
                
                var secretStoreType = Environment.GetEnvironmentVariable("SECRET_STORE_TYPE") ?? "environment";
                
                switch (secretStoreType.ToLowerInvariant())
                {
                    case "azurekeyvault":
                        ValidateAzureKeyVault(result);
                        break;
                    case "hashicorpvault":
                        ValidateHashiCorpVault(result);
                        break;
                    case "environment":
                        // Environment variables are always accessible if process has permissions
                        result.AddValidatedSecret("EnvironmentVariables");
                        _logger.LogDebug("✅ [SECRETS] Using environment variables for secrets");
                        break;
                    default:
                        result.AddError($"Unknown secret store type: {secretStoreType}");
                        _logger.LogError("🚨 [SECRETS] Unknown secret store type: {Type}", secretStoreType);
                        break;
                }
            }
            catch (InvalidOperationException ex)
            {
                result.AddError($"Invalid operation validating secret store: {ex.Message}");
                _logger.LogError(ex, "🚨 [SECRETS] Invalid operation validating secret store access");
            }
            catch (UnauthorizedAccessException ex)
            {
                result.AddError($"Access denied to secret store: {ex.Message}");
                _logger.LogError(ex, "🚨 [SECRETS] Access denied validating secret store access");
            }
            catch (ArgumentException ex)
            {
                result.AddError($"Invalid argument for secret store: {ex.Message}");
                _logger.LogError(ex, "🚨 [SECRETS] Invalid argument validating secret store access");
            }
        }

        private void ValidateAzureKeyVault(StartupValidationResult result)
        {
            // Placeholder for Azure Key Vault validation
            var keyVaultUrl = Environment.GetEnvironmentVariable("AZURE_KEYVAULT_URL");
            if (string.IsNullOrEmpty(keyVaultUrl))
            {
                result.AddMissingSecret("AZURE_KEYVAULT_URL");
            }
            else
            {
                result.AddValidatedSecret("AzureKeyVault");
                _logger.LogDebug("✅ [SECRETS] Azure Key Vault configured");
            }
        }

        private void ValidateHashiCorpVault(StartupValidationResult result)
        {
            // Placeholder for HashiCorp Vault validation
            var vaultUrl = Environment.GetEnvironmentVariable("VAULT_ADDR");
            var vaultToken = Environment.GetEnvironmentVariable("VAULT_TOKEN");
            
            if (string.IsNullOrEmpty(vaultUrl) || string.IsNullOrEmpty(vaultToken))
            {
                if (string.IsNullOrEmpty(vaultUrl)) result.AddMissingSecret("VAULT_ADDR");
                if (string.IsNullOrEmpty(vaultToken)) result.AddMissingSecret("VAULT_TOKEN");
            }
            else
            {
                result.AddValidatedSecret("HashiCorpVault");
                _logger.LogDebug("✅ [SECRETS] HashiCorp Vault configured");
            }
        }

        private static bool IsValidApiKey(string keyName, string keyValue)
        {
            // Basic API key format validation
            return keyName switch
            {
                "TOPSTEPX_API_KEY" => keyValue.Length >= TopstepXApiKeyMinLength && keyValue.All(c => char.IsLetterOrDigit(c) || c == '-'),
                "TOPSTEPX_API_SECRET" => keyValue.Length >= TopstepXApiSecretMinLength,
                _ => keyValue.Length >= DefaultApiKeyMinLength
            };
        }

        private static bool IsHardcodedEndpoint(string endpoint)
        {
            // Check for localhost, development, or other non-production patterns
            var hardcodedPatterns = new[]
            {
                @"localhost",
                @"127\.0\.0\.1",
                @"\.local$",
                @"dev\.|development\.",
                @"test\.|testing\.",
                @"example\.com"
            };

            return hardcodedPatterns.Any(pattern => Regex.IsMatch(endpoint, pattern, RegexOptions.IgnoreCase));
        }

        private static bool ContainsHardcodedCredentials(string connectionString)
        {
            // Check for hardcoded passwords or sensitive data in connection strings
            var sensitivePatterns = new[]
            {
                @"password=(?!{|\$)[^;]+",
                @"pwd=(?!{|\$)[^;]+",
                @"user\s*id=(?!{|\$)[^;]+sa",
                @"uid=(?!{|\$)[^;]+sa"
            };

            return sensitivePatterns.Any(pattern => Regex.IsMatch(connectionString, pattern, RegexOptions.IgnoreCase));
        }

        private static string MaskSensitiveValue(string value)
        {
            if (string.IsNullOrEmpty(value)) return value;
            
            return value.Length <= 8 
                ? new string('*', value.Length)
                : value[..SecretPrefixLength] + new string('*', value.Length - SecretMaskingOverhead) + value[^SecretSuffixLength..];
        }

        private static List<SecretValidationRule> InitializeValidationRules()
        {
            return new List<SecretValidationRule>
            {
                new() { Name = "API Key Pattern", Pattern = new Regex(@"[A-Za-z0-9]{32,}", RegexOptions.Compiled) },
                new() { Name = "Password Pattern", Pattern = new Regex(@"password.*[=:]\s*['""][^'""]{8,}['""]", RegexOptions.IgnoreCase | RegexOptions.Compiled) },
                new() { Name = "Secret Token Pattern", Pattern = new Regex(@"(secret|token|key).*[=:]\s*['""][^'""]{16,}['""]", RegexOptions.IgnoreCase | RegexOptions.Compiled) },
                new() { Name = "JWT Token Pattern", Pattern = new Regex(@"eyJ[A-Za-z0-9-_]+\.eyJ[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+", RegexOptions.Compiled) }
            };
        }
    }

    /// <summary>
    /// Secret validation rule
    /// </summary>
    public class SecretValidationRule
    {
        public string Name { get; set; } = string.Empty;
        public Regex Pattern { get; set; } = new(".*");
    }

    /// <summary>
    /// Result of startup secrets validation
    /// </summary>
    public class StartupValidationResult
    {
        // Private backing fields for collections
        private readonly List<string> _validatedSecrets = new();
        private readonly List<string> _missingSecrets = new();
        private readonly List<string> _invalidSecrets = new();
        private readonly List<string> _errors = new();
        
        public DateTime ValidationTime { get; set; }
        public bool IsValid { get; set; }
        
        // Read-only collection properties
        public IReadOnlyList<string> ValidatedSecrets => _validatedSecrets;
        public IReadOnlyList<string> MissingSecrets => _missingSecrets;
        public IReadOnlyList<string> InvalidSecrets => _invalidSecrets;
        public IReadOnlyList<string> Errors => _errors;
        
        // Replace methods for controlled mutation
        public void ReplaceValidatedSecrets(IEnumerable<string> items)
        {
            _validatedSecrets.Clear();
            if (items != null) _validatedSecrets.AddRange(items);
        }
        
        public void ReplaceMissingSecrets(IEnumerable<string> items)
        {
            _missingSecrets.Clear();
            if (items != null) _missingSecrets.AddRange(items);
        }
        
        public void ReplaceInvalidSecrets(IEnumerable<string> items)
        {
            _invalidSecrets.Clear();
            if (items != null) _invalidSecrets.AddRange(items);
        }
        
        public void ReplaceErrors(IEnumerable<string> items)
        {
            _errors.Clear();
            if (items != null) _errors.AddRange(items);
        }
        
        // Add methods for individual items
        public void AddValidatedSecret(string secret)
        {
            if (!string.IsNullOrEmpty(secret)) _validatedSecrets.Add(secret);
        }
        
        public void AddMissingSecret(string secret)
        {
            if (!string.IsNullOrEmpty(secret)) _missingSecrets.Add(secret);
        }
        
        public void AddInvalidSecret(string secret)
        {
            if (!string.IsNullOrEmpty(secret)) _invalidSecrets.Add(secret);
        }
        
        public void AddError(string error)
        {
            if (!string.IsNullOrEmpty(error)) _errors.Add(error);
        }
    }
}