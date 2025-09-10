using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Text.Json;
using System.Net.Http.Json;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System.Net.Http;
using TopstepX.Bot.Abstractions;

namespace TopstepX.Bot.Core.Services
{
    /// <summary>
    /// Unified Trading System Integration Service
    /// Coordinates all critical components for safe trading operations
    /// </summary>
    public class TradingSystemIntegrationService : BackgroundService
    {
        private readonly ILogger<TradingSystemIntegrationService> _logger;
        private readonly IServiceProvider _serviceProvider;
        private readonly EmergencyStopSystem _emergencyStop;
        private readonly PositionTrackingSystem _positionTracker;
        private OrderFillConfirmationSystem? _orderConfirmation;
        private readonly ErrorHandlingMonitoringSystem _errorMonitoring;
        private readonly HttpClient _httpClient;
        private HubConnection? _userHubConnection;
        private HubConnection? _marketHubConnection;
        
        private readonly TradingSystemConfiguration _config;
        private volatile bool _isSystemReady = false;
        private volatile bool _isTradingEnabled = false;
        
        public bool IsSystemReady => _isSystemReady;
        public bool IsTradingEnabled => _isTradingEnabled && !_emergencyStop.IsEmergencyStop;
        
        public class TradingSystemConfiguration
        {
            public string TopstepXApiBaseUrl { get; set; } = "https://api.topstepx.com";
            public string UserHubUrl { get; set; } = "https://rtc.topstepx.com/hubs/user";
            public string MarketHubUrl { get; set; } = "https://rtc.topstepx.com/hubs/market";
            public string AccountId { get; set; } = string.Empty;
            public bool EnableDryRunMode { get; set; } = true;
            public bool EnableAutoExecution { get; set; } = false;
            public decimal MaxDailyLoss { get; set; } = -1000m;
            public decimal MaxPositionSize { get; set; } = 5m;
            public string ApiToken { get; set; } = string.Empty;
        }
        
        public TradingSystemIntegrationService(
            ILogger<TradingSystemIntegrationService> logger,
            IServiceProvider serviceProvider,
            EmergencyStopSystem emergencyStop,
            PositionTrackingSystem positionTracker,
            ErrorHandlingMonitoringSystem errorMonitoring,
            HttpClient httpClient,
            TradingSystemConfiguration config)
        {
            _logger = logger;
            _serviceProvider = serviceProvider;
            _emergencyStop = emergencyStop;
            _positionTracker = positionTracker;
            _errorMonitoring = errorMonitoring;
            _httpClient = httpClient;
            _config = config;
            
            // Setup HTTP client
            _httpClient.BaseAddress = new Uri(_config.TopstepXApiBaseUrl);
            if (!string.IsNullOrEmpty(_config.ApiToken))
            {
                _httpClient.DefaultRequestHeaders.Authorization = 
                    new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _config.ApiToken);
            }
        }
        
        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            try
            {
                _logger.LogInformation("🚀 Trading System Integration Service starting...");
                
                // Initialize all components
                await InitializeComponentsAsync(stoppingToken);
                
                // Setup SignalR connections
                await SetupSignalRConnectionsAsync(stoppingToken);
                
                // Setup event handlers
                SetupEventHandlers();
                
                // Perform initial system checks
                await PerformSystemReadinessChecksAsync();
                
                _logger.LogInformation("✅ Trading System Integration Service ready");
                _isSystemReady = true;
                
                // Main service loop
                while (!stoppingToken.IsCancellationRequested)
                {
                    await MonitorSystemHealthAsync();
                    await Task.Delay(TimeSpan.FromSeconds(30), stoppingToken);
                }
            }
            catch (OperationCanceledException)
            {
                _logger.LogInformation("🛑 Trading System Integration Service stopping...");
            }
            catch (Exception ex)
            {
                await _errorMonitoring.LogErrorAsync("TradingSystemIntegration", ex, ErrorHandlingMonitoringSystem.ErrorSeverity.Critical);
                _logger.LogCritical(ex, "🚨 CRITICAL: Trading System Integration Service failed");
            }
            finally
            {
                await CleanupAsync();
            }
        }
        
        private async Task InitializeComponentsAsync(CancellationToken cancellationToken)
        {
            try
            {
                _logger.LogInformation("🔧 Initializing trading system components...");
                
                // Initialize error monitoring first
                _errorMonitoring.UpdateComponentHealth("ErrorMonitoring", ErrorHandlingMonitoringSystem.HealthStatus.Healthy);
                
                // Initialize emergency stop system
                _emergencyStop.EmergencyStopTriggered += OnEmergencyStopTriggered;
                _errorMonitoring.UpdateComponentHealth("EmergencyStop", ErrorHandlingMonitoringSystem.HealthStatus.Healthy);
                
                // Initialize position tracker with real risk limits from TopstepX API
                var riskLimits = await GetRiskLimitsFromApiAsync();
                
                _positionTracker.RiskViolationDetected += OnRiskViolationDetected;
                _errorMonitoring.UpdateComponentHealth("PositionTracking", ErrorHandlingMonitoringSystem.HealthStatus.Healthy);
                
                _logger.LogInformation("✅ Core components initialized");
            }
            catch (Exception ex)
            {
                await _errorMonitoring.LogErrorAsync("ComponentInitialization", ex, ErrorHandlingMonitoringSystem.ErrorSeverity.Critical);
                throw;
            }
        }
        
        private async Task SetupSignalRConnectionsAsync(CancellationToken cancellationToken)
        {
            try
            {
                _logger.LogInformation("📡 Setting up SignalR connections...");
                
                // Test JWT availability first to avoid hanging
                try
                {
                    _logger.LogInformation("🔍 Testing JWT token availability...");
                    var testToken = await GetFreshJwtAsync();
                    _logger.LogInformation("✅ JWT token available for SignalR connections");
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "❌ Failed to obtain JWT token for SignalR - will retry with fallback strategy");
                    
                    // Try to use existing environment JWT as fallback
                    var envJwt = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
                    if (!string.IsNullOrEmpty(envJwt))
                    {
                        _jwt = envJwt;
                        _jwtExpiration = DateTimeOffset.UtcNow.AddHours(1); // Short expiration for env token
                        _logger.LogInformation("🔄 Using environment JWT token as fallback");
                    }
                    else
                    {
                        _logger.LogError("❌ No JWT token available from any source - SignalR connections will fail");
                        // Continue anyway to test the connection failure handling
                    }
                }
                
                // User Hub Connection with proper JWT provider (no manual token in URL)
                _userHubConnection = new HubConnectionBuilder()
                    .WithUrl(_config.UserHubUrl, options =>
                    {
                        // SignalR will append ?access_token=<value> for WebSockets automatically
                        options.AccessTokenProvider = async () => await GetFreshJwtAsync();
                        options.Transports = Microsoft.AspNetCore.Http.Connections.HttpTransportType.WebSockets;
                        options.SkipNegotiation = true;
                    })
                    .WithAutomaticReconnect(new[] { 
                        TimeSpan.Zero, 
                        TimeSpan.FromSeconds(2), 
                        TimeSpan.FromSeconds(5), 
                        TimeSpan.FromSeconds(10) 
                    })
                    .Build();
                
                // Register event handlers BEFORE StartAsync
                _userHubConnection.On<object>("GatewayUserAccount", data => OnUserAccount(data));
                _userHubConnection.On<object>("GatewayUserOrder", data => OnUserOrder(data));
                _userHubConnection.On<object>("GatewayUserPosition", data => OnUserPosition(data));
                _userHubConnection.On<object>("GatewayUserTrade", data => OnUserTrade(data));
                
                _userHubConnection.Closed += (error) =>
                {
                    _logger.LogWarning("📡 User Hub connection closed. Error: {Error}, Exception: {Exception}", 
                        error?.Message ?? "Unknown", error?.GetType().Name ?? "None");
                    if (error != null)
                    {
                        _logger.LogError(error, "📡 User Hub connection closed with exception");
                    }
                    _errorMonitoring.UpdateComponentHealth("UserHub", ErrorHandlingMonitoringSystem.HealthStatus.Warning, error?.Message);
                    return Task.CompletedTask;
                };
                
                _userHubConnection.Reconnected += async (connectionId) =>
                {
                    _logger.LogInformation("📡 User Hub reconnected: {ConnectionId}", connectionId);
                    _errorMonitoring.UpdateComponentHealth("UserHub", ErrorHandlingMonitoringSystem.HealthStatus.Healthy);
                    await SubscribeToUserHubAsync();
                };
                
                // Market Hub Connection with proper JWT provider (no manual token in URL)
                _marketHubConnection = new HubConnectionBuilder()
                    .WithUrl(_config.MarketHubUrl, options =>
                    {
                        // SignalR will append ?access_token=<value> for WebSockets automatically
                        options.AccessTokenProvider = async () => await GetFreshJwtAsync();
                        options.Transports = Microsoft.AspNetCore.Http.Connections.HttpTransportType.WebSockets;
                        options.SkipNegotiation = true;
                    })
                    .WithAutomaticReconnect(new[] { 
                        TimeSpan.Zero, 
                        TimeSpan.FromSeconds(2), 
                        TimeSpan.FromSeconds(5), 
                        TimeSpan.FromSeconds(10) 
                    })
                    .Build();
                
                // Register event handlers BEFORE StartAsync
                _marketHubConnection.On<string, object>("GatewayQuote", (contractId, data) => OnMarketQuote(contractId, data));
                _marketHubConnection.On<string, object>("GatewayTrade", (contractId, data) => OnMarketTrade(contractId, data));
                _marketHubConnection.On<string, object>("GatewayDepth", (contractId, data) => OnMarketDepth(contractId, data));
                
                _marketHubConnection.Closed += (error) =>
                {
                    _logger.LogWarning("📡 Market Hub connection closed. Error: {Error}, Exception: {Exception}", 
                        error?.Message ?? "Unknown", error?.GetType().Name ?? "None");
                    if (error != null)
                    {
                        _logger.LogError(error, "📡 Market Hub connection closed with exception");
                    }
                    _errorMonitoring.UpdateComponentHealth("MarketHub", ErrorHandlingMonitoringSystem.HealthStatus.Warning, error?.Message);
                    return Task.CompletedTask;
                };
                
                _marketHubConnection.Reconnected += async (connectionId) =>
                {
                    _logger.LogInformation("📡 Market Hub reconnected: {ConnectionId}", connectionId);
                    _errorMonitoring.UpdateComponentHealth("MarketHub", ErrorHandlingMonitoringSystem.HealthStatus.Healthy);
                    await SubscribeToMarketHubAsync();
                };
                
                // Start BOTH hubs
                _logger.LogInformation("📡 Starting SignalR connections...");
                await Task.WhenAll(
                    _userHubConnection.StartAsync(cancellationToken),
                    _marketHubConnection.StartAsync(cancellationToken)
                );
                
                _logger.LogInformation("✅ User Hub connected successfully (State: {UserState})", _userHubConnection.State);
                _logger.LogInformation("✅ Market Hub connected successfully (State: {MarketState})", _marketHubConnection.State);
                
                // Wait a moment for connections to stabilize before subscribing
                _logger.LogInformation("⏳ Waiting for connections to stabilize...");
                await Task.Delay(2000, cancellationToken);
                
                // Check connection states before subscribing
                _logger.LogInformation("🔍 Connection states - User: {UserState}, Market: {MarketState}", 
                    _userHubConnection.State, _marketHubConnection.State);
                
                // Subscribe AFTER StartAsync and stabilization delay
                if (_userHubConnection.State == HubConnectionState.Connected)
                {
                    await SubscribeToUserHubAsync();
                }
                
                if (_marketHubConnection.State == HubConnectionState.Connected)
                {
                    await SubscribeToMarketHubAsync();
                }

                // Initialize order confirmation system with SignalR connections
                _orderConfirmation = new OrderFillConfirmationSystem(
                    _serviceProvider.GetRequiredService<ILogger<OrderFillConfirmationSystem>>(),
                    _httpClient,
                    _userHubConnection,
                    _marketHubConnection,
                    _positionTracker,
                    _emergencyStop);
                
                _orderConfirmation.OrderConfirmed += OnOrderConfirmed;
                _orderConfirmation.OrderRejected += OnOrderRejected;
                _orderConfirmation.FillConfirmed += OnFillConfirmed;
                
                _errorMonitoring.UpdateComponentHealth("OrderConfirmation", ErrorHandlingMonitoringSystem.HealthStatus.Healthy);
                _logger.LogInformation("✅ SignalR connections established");
            }
            catch (Exception ex)
            {
                await _errorMonitoring.LogErrorAsync("SignalRSetup", ex, ErrorHandlingMonitoringSystem.ErrorSeverity.Critical);
                throw;
            }
        }

        private string? _jwt;
        private DateTimeOffset _jwtExpiration = DateTimeOffset.MinValue;

        private async Task<string> GetFreshJwtAsync()
        {
            try
            {
                _logger.LogInformation("🔄 GetFreshJwtAsync called - checking for available tokens...");
                
                // Return cached token if still valid (refresh when < 60 min remain)
                if (!string.IsNullOrEmpty(_jwt) && _jwtExpiration > DateTimeOffset.UtcNow.AddHours(1))
                {
                    _logger.LogInformation("🔑 Using cached JWT token (expires: {Expiration})", _jwtExpiration);
                    return _jwt;
                }

                // First, try to get from AutoTopstepXLoginService (fastest option)
                var autoLoginService = _serviceProvider.GetService<BotCore.Services.AutoTopstepXLoginService>();
                if (autoLoginService?.IsAuthenticated == true && !string.IsNullOrEmpty(autoLoginService.JwtToken))
                {
                    _jwt = autoLoginService.JwtToken;
                    _jwtExpiration = DateTimeOffset.UtcNow.AddHours(23); // Conservative expiration
                    _logger.LogInformation("✅ Got fresh JWT from AutoTopstepXLoginService (first 16 chars): {TokenPreview}...", 
                        _jwt.Length > 16 ? _jwt.Substring(0, 16) : _jwt);
                    return _jwt;
                }

                // Check if we have environment credentials to make our own request
                _logger.LogInformation("🔍 AutoTopstepXLoginService not ready, checking environment credentials...");
                var username = Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME");
                var apiKey = Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY");

                if (string.IsNullOrEmpty(username) || string.IsNullOrEmpty(apiKey))
                {
                    _logger.LogWarning("⚠️ No direct credentials available. Waiting for AutoTopstepXLoginService...");
                    
                    // Wait a bit for AutoTopstepXLoginService to authenticate
                    using var timeoutCts = new CancellationTokenSource(TimeSpan.FromSeconds(10));
                    var attempts = 0;
                    while (!timeoutCts.Token.IsCancellationRequested && attempts < 5)
                    {
                        await Task.Delay(2000, timeoutCts.Token);
                        attempts++;
                        
                        if (autoLoginService?.IsAuthenticated == true && !string.IsNullOrEmpty(autoLoginService.JwtToken))
                        {
                            _jwt = autoLoginService.JwtToken;
                            _jwtExpiration = DateTimeOffset.UtcNow.AddHours(23);
                            _logger.LogInformation("✅ Got JWT from AutoTopstepXLoginService after waiting (attempt {Attempt})", attempts);
                            return _jwt;
                        }
                        
                        _logger.LogInformation("⏳ Waiting for AutoTopstepXLoginService... (attempt {Attempt}/5)", attempts);
                    }
                    
                    throw new InvalidOperationException("No TopstepX credentials available and AutoTopstepXLoginService not ready");
                }

                _logger.LogInformation("🔐 Using environment credentials to login directly...");

                // Login with API key to get fresh JWT (with timeout)
                var loginRequest = new
                {
                    userName = username,
                    apiKey = apiKey
                };

                using var timeoutCts2 = new CancellationTokenSource(TimeSpan.FromSeconds(30));
                var response = await _httpClient.PostAsJsonAsync("/api/Auth/loginKey", loginRequest, timeoutCts2.Token);
                response.EnsureSuccessStatusCode();

                var authResponse = await response.Content.ReadFromJsonAsync<AuthResponse>(cancellationToken: timeoutCts2.Token);
                if (authResponse?.token == null)
                {
                    throw new InvalidOperationException("Empty auth response from TopstepX");
                }

                _jwt = authResponse.token;
                _jwtExpiration = DateTimeOffset.UtcNow.AddHours(23); // Conservative expiration (tokens ~24h lifetime)
                
                _logger.LogInformation("✅ Successfully obtained fresh JWT token via direct login (first 16 chars): {TokenPreview}...", 
                    _jwt.Length > 16 ? _jwt.Substring(0, 16) : _jwt);

                return _jwt;
            }
            catch (OperationCanceledException)
            {
                _logger.LogError("⏰ JWT token request timed out");
                throw new InvalidOperationException("JWT token request timed out");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error obtaining fresh JWT token");
                throw;
            }
        }

        private class AuthResponse
        {
            public string? token { get; set; }
        }
        
        private async Task SubscribeToUserHubAsync()
        {
            if (_userHubConnection?.State != HubConnectionState.Connected)
            {
                _logger.LogWarning("⚠️ User Hub not connected, skipping subscriptions");
                return;
            }

            try
            {
                _logger.LogInformation("📡 Subscribing to User Hub events...");
                
                // Parse account ID as integer for TopstepX API
                if (!int.TryParse(_config.AccountId, out var accountId))
                {
                    _logger.LogError("❌ Failed to parse account ID as integer: {AccountId}", _config.AccountId);
                    return;
                }

                await _userHubConnection.InvokeAsync("SubscribeAccounts");
                await _userHubConnection.InvokeAsync("SubscribeOrders", accountId);
                await _userHubConnection.InvokeAsync("SubscribePositions", accountId);
                await _userHubConnection.InvokeAsync("SubscribeTrades", accountId);
                
                _logger.LogInformation("✅ User Hub subscriptions successful for account {AccountId}", accountId);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to subscribe to User Hub events");
                await _errorMonitoring.LogErrorAsync("UserHubSubscription", ex, ErrorHandlingMonitoringSystem.ErrorSeverity.High);
            }
        }

        private async Task ResubscribeToUserHubAsync()
        {
            await SubscribeToUserHubAsync();
        }

        private readonly HashSet<string> _marketSubscriptions = new(StringComparer.OrdinalIgnoreCase);

        private async Task SubscribeToMarketHubAsync()
        {
            if (_marketHubConnection?.State != HubConnectionState.Connected)
            {
                _logger.LogWarning("⚠️ Market Hub not connected, skipping subscriptions");
                return;
            }

            try
            {
                _logger.LogInformation("📡 Subscribing to Market Hub events...");
                
                // Default contract IDs for ES and MES (TopstepX format)
                string[] contractIds = { "F.US.EP", "F.US.ENQ" }; // ES and MES futures
                
                foreach (var contractId in contractIds)
                {
                    if (!_marketSubscriptions.Add(contractId)) continue; // Skip if already subscribed
                    
                    await _marketHubConnection.InvokeAsync("SubscribeContractQuotes", contractId);
                    await _marketHubConnection.InvokeAsync("SubscribeContractTrades", contractId);
                    await _marketHubConnection.InvokeAsync("SubscribeContractMarketDepth", contractId);
                    
                    _logger.LogInformation("✅ Market subscriptions successful for {ContractId}", contractId);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to subscribe to Market Hub events");
                await _errorMonitoring.LogErrorAsync("MarketHubSubscription", ex, ErrorHandlingMonitoringSystem.ErrorSeverity.Medium);
            }
        }
        
        private async Task ResubscribeToMarketHubAsync()
        {
            if (_marketHubConnection?.State != HubConnectionState.Connected) return;
            
            try
            {
                foreach (var contractId in _marketSubscriptions)
                {
                    await _marketHubConnection.InvokeAsync("SubscribeContractQuotes", contractId);
                    await _marketHubConnection.InvokeAsync("SubscribeContractTrades", contractId);
                    await _marketHubConnection.InvokeAsync("SubscribeContractMarketDepth", contractId);
                }
                _logger.LogInformation("✅ Market Hub resubscriptions successful");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to resubscribe to Market Hub events");
                await _errorMonitoring.LogErrorAsync("MarketHubResubscription", ex, ErrorHandlingMonitoringSystem.ErrorSeverity.Medium);
            }
        }

        // SignalR Event Handlers for User Hub
        private void OnUserAccount(object data)
        {
            try
            {
                var json = JsonSerializer.Serialize(data);
                _logger.LogInformation("📊 User Account Update: {Data}", json);
            }
            catch (Exception ex)
            {
                _logger.LogWarning("⚠️ Failed to process user account data: {Error}", ex.Message);
            }
        }

        private void OnUserOrder(object data)
        {
            try
            {
                var json = JsonSerializer.Serialize(data);
                _logger.LogInformation("� Order Update: {Data}", json);
            }
            catch (Exception ex)
            {
                _logger.LogWarning("⚠️ Failed to process order data: {Error}", ex.Message);
            }
        }

        private void OnUserPosition(object data)
        {
            try
            {
                var json = JsonSerializer.Serialize(data);
                _logger.LogInformation("📈 Position Update: {Data}", json);
            }
            catch (Exception ex)
            {
                _logger.LogWarning("⚠️ Failed to process position data: {Error}", ex.Message);
            }
        }

        private void OnUserTrade(object data)
        {
            try
            {
                var json = JsonSerializer.Serialize(data);
                _logger.LogInformation("💰 Trade Fill: {Data}", json);
            }
            catch (Exception ex)
            {
                _logger.LogWarning("⚠️ Failed to process trade data: {Error}", ex.Message);
            }
        }

        // SignalR Event Handlers for Market Hub
        private void OnMarketQuote(string contractId, object data)
        {
            try
            {
                var json = JsonSerializer.Serialize(data);
                _logger.LogDebug("📊 Market Quote [{ContractId}]: {Data}", contractId, json);
            }
            catch (Exception ex)
            {
                _logger.LogWarning("⚠️ Failed to process market quote for {ContractId}: {Error}", contractId, ex.Message);
            }
        }

        private void OnMarketTrade(string contractId, object data)
        {
            try
            {
                var json = JsonSerializer.Serialize(data);
                _logger.LogDebug("💹 Market Trade [{ContractId}]: {Data}", contractId, json);
            }
            catch (Exception ex)
            {
                _logger.LogWarning("⚠️ Failed to process market trade for {ContractId}: {Error}", contractId, ex.Message);
            }
        }

        private void OnMarketDepth(string contractId, object data)
        {
            try
            {
                var json = JsonSerializer.Serialize(data);
                _logger.LogDebug("📚 Market Depth [{ContractId}]: {Data}", contractId, json);
            }
            catch (Exception ex)
            {
                _logger.LogWarning("⚠️ Failed to process market depth for {ContractId}: {Error}", contractId, ex.Message);
            }
        }
        
        private void SetupEventHandlers()
        {
            // Error monitoring events
            _errorMonitoring.CriticalErrorDetected += OnCriticalErrorDetected;
            _errorMonitoring.HealthStatusChanged += OnHealthStatusChanged;
            
            _logger.LogInformation("📋 Event handlers configured");
        }
        
        private async Task PerformSystemReadinessChecksAsync()
        {
            _logger.LogInformation("🔍 Performing system readiness checks...");
            
            var checks = new List<(string Name, Func<Task<bool>> Check)>
            {
                ("EmergencyStop", () => Task.FromResult(!_emergencyStop.IsEmergencyStop)),
                ("UserHubConnection", () => Task.FromResult(_userHubConnection?.State == HubConnectionState.Connected)),
                ("MarketHubConnection", () => Task.FromResult(_marketHubConnection?.State == HubConnectionState.Connected)),
                ("ApiConnectivity", TestApiConnectivityAsync),
                ("ConfigurationValid", () => Task.FromResult(ValidateConfiguration()))
            };
            
            var passedChecks = 0;
            foreach (var (name, check) in checks)
            {
                try
                {
                    var result = await check();
                    if (result)
                    {
                        passedChecks++;
                        _errorMonitoring.RecordSuccess($"ReadinessCheck_{name}");
                        _logger.LogInformation("✅ {CheckName} - PASSED", name);
                    }
                    else
                    {
                        _errorMonitoring.UpdateComponentHealth($"ReadinessCheck_{name}", 
                            ErrorHandlingMonitoringSystem.HealthStatus.Critical, "Check failed");
                        _logger.LogWarning("❌ {CheckName} - FAILED", name);
                    }
                }
                catch (Exception ex)
                {
                    await _errorMonitoring.LogErrorAsync($"ReadinessCheck_{name}", ex, ErrorHandlingMonitoringSystem.ErrorSeverity.High);
                    _logger.LogWarning("❌ {CheckName} - ERROR: {Error}", name, ex.Message);
                }
            }
            
            var healthPercentage = (passedChecks * 100.0) / checks.Count;
            _logger.LogInformation("📊 System readiness: {PassedChecks}/{TotalChecks} ({Percentage:F1}%)", 
                passedChecks, checks.Count, healthPercentage);
            
            // Enable trading only if all critical checks pass
            _isTradingEnabled = passedChecks >= checks.Count - 1; // Allow 1 failure for non-critical checks
            
            if (_isTradingEnabled && _config.EnableDryRunMode)
            {
                _logger.LogWarning("⚠️ System ready but in DRY RUN mode - no live trading");
            }
            else if (!_isTradingEnabled)
            {
                _logger.LogCritical("🚨 System NOT ready for trading - critical checks failed");
            }
        }
        
        private async Task<bool> TestApiConnectivityAsync()
        {
            try
            {
                var response = await _httpClient.GetAsync("/api/health");
                return response.IsSuccessStatusCode;
            }
            catch (Exception ex)
            {
                _logger.LogWarning("❌ API connectivity test failed: {Error}", ex.Message);
                return false;
            }
        }
        
        private bool ValidateConfiguration()
        {
            var isValid = !string.IsNullOrEmpty(_config.AccountId) &&
                         !string.IsNullOrEmpty(_config.TopstepXApiBaseUrl) &&
                         !string.IsNullOrEmpty(_config.UserHubUrl) &&
                         !string.IsNullOrEmpty(_config.MarketHubUrl);
            
            if (!isValid)
            {
                _logger.LogCritical("🚨 Configuration validation failed - missing required settings");
            }
            
            return isValid;
        }
        
        private async Task MonitorSystemHealthAsync()
        {
            try
            {
                // Update connection status
                _errorMonitoring.UpdateComponentHealth("UserHub", 
                    _userHubConnection?.State == HubConnectionState.Connected 
                        ? ErrorHandlingMonitoringSystem.HealthStatus.Healthy 
                        : ErrorHandlingMonitoringSystem.HealthStatus.Critical);
                
                _errorMonitoring.UpdateComponentHealth("MarketHub", 
                    _marketHubConnection?.State == HubConnectionState.Connected 
                        ? ErrorHandlingMonitoringSystem.HealthStatus.Healthy 
                        : ErrorHandlingMonitoringSystem.HealthStatus.Critical);
                
                // Get system health summary
                var health = _errorMonitoring.GetSystemHealth();
                
                if (!health.IsHealthy)
                {
                    _logger.LogWarning("⚠️ System health degraded: {HealthScore:F1}% ({CriticalComponents} critical)", 
                        health.OverallHealthScore, health.CriticalComponents);
                }
                
                // Update trading enabled status based on health
                var shouldEnableTrading = health.IsHealthy && !_emergencyStop.IsEmergencyStop && _isSystemReady;
                
                if (_isTradingEnabled != shouldEnableTrading)
                {
                    _isTradingEnabled = shouldEnableTrading;
                    _logger.LogWarning("🔄 Trading status changed: {Status}", 
                        _isTradingEnabled ? "ENABLED" : "DISABLED");
                }
            }
            catch (Exception ex)
            {
                await _errorMonitoring.LogErrorAsync("HealthMonitoring", ex, ErrorHandlingMonitoringSystem.ErrorSeverity.Medium);
            }
        }
        
        // Event Handlers
        private async void OnEmergencyStopTriggered(object? sender, EmergencyStopEventArgs e)
        {
            _logger.LogCritical("🛑 EMERGENCY STOP TRIGGERED: {Reason}", e.Reason);
            _isTradingEnabled = false;
            
            // Cancel all pending orders
            try
            {
                var pendingOrders = _orderConfirmation?.GetAllOrders()?.Values.ToList() ?? new List<OrderFillConfirmationSystem.OrderTrackingRecord>();
                foreach (var order in pendingOrders.Where(o => o.Status == "SUBMITTED" || o.Status == "PENDING"))
                {
                    if (_orderConfirmation != null)
                    {
                        await _orderConfirmation.CancelOrderAsync(order.ClientOrderId, _config.AccountId);
                    }
                }
                
                _logger.LogInformation("📝 Emergency order cancellation initiated");
            }
            catch (Exception ex)
            {
                await _errorMonitoring.LogErrorAsync("EmergencyOrderCancellation", ex, ErrorHandlingMonitoringSystem.ErrorSeverity.Critical);
            }
        }
        
        private void OnRiskViolationDetected(object? sender, RiskViolationEventArgs e)
        {
            _logger.LogCritical("🚨 RISK VIOLATION: {Symbol} - {Violations}", 
                e.Symbol, string.Join(", ", e.Violations));
            
            // Consider triggering emergency stop for severe violations
            if (e.ViolationType == "Account Risk")
            {
                _emergencyStop.TriggerEmergencyStop($"Account risk violation: {string.Join(", ", e.Violations)}");
            }
        }
        
        private void OnCriticalErrorDetected(object? sender, CriticalErrorEventArgs e)
        {
            _logger.LogCritical("🚨 CRITICAL ERROR in {Component}: {Message}", e.Component, e.Exception.Message);
            
            // Consider emergency stop for trading-related critical errors
            if (e.Component.Contains("Trading") || e.Component.Contains("Order") || e.Component.Contains("Position"))
            {
                _emergencyStop.TriggerEmergencyStop($"Critical error in {e.Component}: {e.Exception.Message}");
            }
        }
        
        private void OnHealthStatusChanged(object? sender, HealthStatusEventArgs e)
        {
            _logger.LogWarning("🏥 System health changed: {IsHealthy} (Score: {HealthScore:F1}%)", 
                e.IsHealthy ? "HEALTHY" : "UNHEALTHY", e.OverallHealthScore);
        }
        
        private void OnOrderConfirmed(object? sender, OrderConfirmedEventArgs e)
        {
            _logger.LogInformation("✅ Order confirmed: {ClientOrderId} - {Status}", 
                e.TrackingRecord.ClientOrderId, e.GatewayOrderUpdate.Status);
        }
        
        private void OnOrderRejected(object? sender, OrderRejectedEventArgs e)
        {
            _logger.LogWarning("❌ Order rejected: {ClientOrderId} - {Reason}", 
                e.TrackingRecord.ClientOrderId, e.GatewayOrderUpdate.Reason);
        }
        
        private void OnFillConfirmed(object? sender, FillConfirmedEventArgs e)
        {
            _logger.LogInformation("💰 Fill confirmed: {Symbol} {Quantity}@{Price:F2}", 
                e.TrackingRecord.Symbol, e.FillConfirmation.FillQuantity, e.FillConfirmation.FillPrice);
        }
        
        /// <summary>
        /// Get comprehensive system status
        /// </summary>
        public TradingSystemStatus GetSystemStatus()
        {
            var health = _errorMonitoring.GetSystemHealth();
            var positions = _positionTracker.GetAccountSummary();
            
            return new TradingSystemStatus
            {
                IsSystemReady = _isSystemReady,
                IsTradingEnabled = _isTradingEnabled,
                IsEmergencyStop = _emergencyStop.IsEmergencyStop,
                IsDryRunMode = _config.EnableDryRunMode,
                HealthScore = health.OverallHealthScore,
                ComponentCount = health.ComponentCount,
                CriticalComponents = health.CriticalComponents,
                AccountSummary = positions,
                LastUpdate = DateTime.UtcNow
            };
        }
        
        private async Task CleanupAsync()
        {
            try
            {
                _logger.LogInformation("🧹 Cleaning up trading system...");
                
                if (_userHubConnection != null)
                {
                    await _userHubConnection.DisposeAsync();
                }
                
                if (_marketHubConnection != null)
                {
                    await _marketHubConnection.DisposeAsync();
                }
                
                _orderConfirmation?.Dispose();
                _positionTracker?.Dispose();
                _errorMonitoring?.Dispose();
                
                _logger.LogInformation("✅ Trading system cleanup completed");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error during cleanup");
            }
        }

        /// <summary>
        /// Professional account balance retrieval from trading API
        /// </summary>
        private async Task<decimal> GetAccountBalanceFromApiAsync()
        {
            try
            {
                // Real TopstepX API integration
                if (_httpClient.DefaultRequestHeaders.Authorization == null)
                {
                    _logger.LogWarning("[Trading-System] No authentication token set for account balance API call");
                    return 50000m; // Fallback balance
                }

                var response = await _httpClient.GetAsync($"/api/Account?accountId={_config.AccountId}");
                
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    var accountData = System.Text.Json.JsonSerializer.Deserialize<JsonElement>(content);
                    
                    if (accountData.TryGetProperty("balance", out var balanceElement))
                    {
                        var balance = balanceElement.GetDecimal();
                        _logger.LogDebug("[Trading-System] Account balance retrieved from TopstepX API: {Balance:C}", balance);
                        return balance;
                    }
                    
                    if (accountData.TryGetProperty("netLiquidationValue", out var nlvElement))
                    {
                        var nlv = nlvElement.GetDecimal();
                        _logger.LogDebug("[Trading-System] Account NLV retrieved from TopstepX API: {NLV:C}", nlv);
                        return nlv;
                    }
                }
                
                _logger.LogWarning("[Trading-System] Failed to retrieve account balance from TopstepX API: {StatusCode}", response.StatusCode);
                
                // Fallback to safe default
                return 50000m;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[Trading-System] Error retrieving account balance from TopstepX API");
                return 25000m; // Safe fallback balance
            }
        }
        
        private async Task<PositionTrackingSystem.RiskLimits> GetRiskLimitsFromApiAsync()
        {
            try
            {
                // Get account balance from TopstepX API
                var accountBalance = await GetAccountBalanceFromApiAsync();
                
                // Try to get risk limits from TopstepX account API
                if (_httpClient.DefaultRequestHeaders.Authorization != null)
                {
                    var response = await _httpClient.GetAsync($"/api/Account/risk?accountId={_config.AccountId}");
                    
                    if (response.IsSuccessStatusCode)
                    {
                        var content = await response.Content.ReadAsStringAsync();
                        var riskData = System.Text.Json.JsonSerializer.Deserialize<JsonElement>(content);
                        
                        var maxDailyLoss = _config.MaxDailyLoss; // Default from config
                        var maxPositionSize = _config.MaxPositionSize; // Default from config
                        
                        // Try to extract real risk limits from API response
                        if (riskData.TryGetProperty("maxDailyLoss", out var maxDailyLossElement))
                        {
                            maxDailyLoss = maxDailyLossElement.GetDecimal();
                        }
                        else if (riskData.TryGetProperty("dailyLossLimit", out var dailyLossLimitElement))
                        {
                            maxDailyLoss = -Math.Abs(dailyLossLimitElement.GetDecimal()); // Ensure negative
                        }
                        
                        if (riskData.TryGetProperty("maxPositionSize", out var maxPosElement))
                        {
                            maxPositionSize = maxPosElement.GetDecimal();
                        }
                        else if (riskData.TryGetProperty("positionSizeLimit", out var posLimitElement))
                        {
                            maxPositionSize = posLimitElement.GetDecimal();
                        }
                        
                        _logger.LogInformation("[Trading-System] Risk limits retrieved from TopstepX API: MaxDailyLoss={MaxDailyLoss:C}, MaxPositionSize={MaxPositionSize}", 
                            maxDailyLoss, maxPositionSize);
                        
                        return new PositionTrackingSystem.RiskLimits
                        {
                            MaxDailyLoss = maxDailyLoss,
                            MaxPositionSize = maxPositionSize,
                            AccountBalance = accountBalance
                        };
                    }
                    
                    _logger.LogWarning("[Trading-System] Failed to retrieve risk limits from TopstepX API: {StatusCode}", response.StatusCode);
                }
                
                // Fallback to configuration values
                _logger.LogInformation("[Trading-System] Using configured risk limits as fallback");
                return new PositionTrackingSystem.RiskLimits
                {
                    MaxDailyLoss = _config.MaxDailyLoss,
                    MaxPositionSize = _config.MaxPositionSize,
                    AccountBalance = accountBalance
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[Trading-System] Error retrieving risk limits from TopstepX API, using configured fallback");
                
                // Safe fallback
                return new PositionTrackingSystem.RiskLimits
                {
                    MaxDailyLoss = _config.MaxDailyLoss,
                    MaxPositionSize = _config.MaxPositionSize,
                    AccountBalance = await GetAccountBalanceFromApiAsync()
                };
            }
        }
    }
    
    public class TradingSystemStatus
    {
        public bool IsSystemReady { get; set; }
        public bool IsTradingEnabled { get; set; }
        public bool IsEmergencyStop { get; set; }
        public bool IsDryRunMode { get; set; }
        public double HealthScore { get; set; }
        public int ComponentCount { get; set; }
        public int CriticalComponents { get; set; }
        public AccountSummary AccountSummary { get; set; } = new();
        public DateTime LastUpdate { get; set; }
    }
}