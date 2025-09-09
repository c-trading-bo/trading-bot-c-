using System;
using System.Net.Mail;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using System.Net;

namespace TradingBot.Infrastructure.Alerts
{
    /// <summary>
    /// Production-grade alert service for email and Slack notifications
    /// Supports model health, latency, and deployment event alerting
    /// </summary>
    public class AlertService : IAlertService
    {
        private readonly ILogger<AlertService> _logger;
        private readonly HttpClient _httpClient;
        private readonly string? _slackWebhook;
        private readonly string? _smtpServer;
        private readonly int _smtpPort;
        private readonly bool _useTls;
        private readonly string? _smtpUsername;
        private readonly string? _smtpPassword;
        private readonly string? _emailFrom;
        private readonly string? _emailTo;

        public AlertService(ILogger<AlertService> logger, HttpClient httpClient)
        {
            _logger = logger;
            _httpClient = httpClient;
            
            // Load configuration from environment
            _slackWebhook = Environment.GetEnvironmentVariable("ALERT_SLACK_WEBHOOK");
            _smtpServer = Environment.GetEnvironmentVariable("ALERT_EMAIL_SMTP");
            _smtpPort = int.TryParse(Environment.GetEnvironmentVariable("ALERT_EMAIL_PORT"), out var port) ? port : 587;
            _useTls = (Environment.GetEnvironmentVariable("ALERT_EMAIL_USE_TLS") ?? "true").ToLowerInvariant() is "true" or "1";
            _smtpUsername = Environment.GetEnvironmentVariable("ALERT_EMAIL_USERNAME");
            _smtpPassword = Environment.GetEnvironmentVariable("ALERT_EMAIL_PASSWORD");
            _emailFrom = Environment.GetEnvironmentVariable("ALERT_EMAIL_FROM");
            _emailTo = Environment.GetEnvironmentVariable("ALERT_EMAIL_TO");
            
            _logger.LogInformation("[ALERT] AlertService initialized - Email: {EmailEnabled}, Slack: {SlackEnabled}, SMTP: {SmtpServer}:{SmtpPort}, TLS: {UseTls}",
                !string.IsNullOrEmpty(_smtpServer), !string.IsNullOrEmpty(_slackWebhook), _smtpServer, _smtpPort, _useTls);
        }

        public async Task SendEmailAsync(string subject, string body, AlertSeverity severity = AlertSeverity.Info)
        {
            if (string.IsNullOrEmpty(_smtpServer) || string.IsNullOrEmpty(_emailFrom) || string.IsNullOrEmpty(_emailTo))
            {
                _logger.LogWarning("[ALERT] Email configuration missing - SMTP: {Smtp}, From: {From}, To: {To}",
                    !string.IsNullOrEmpty(_smtpServer), !string.IsNullOrEmpty(_emailFrom), !string.IsNullOrEmpty(_emailTo));
                return;
            }

            try
            {
                using var smtpClient = new SmtpClient(_smtpServer, _smtpPort);
                
                // Configure SMTP authentication if credentials provided
                if (!string.IsNullOrEmpty(_smtpUsername) && !string.IsNullOrEmpty(_smtpPassword))
                {
                    smtpClient.Credentials = new NetworkCredential(_smtpUsername, _smtpPassword);
                }
                
                smtpClient.EnableSsl = _useTls;
                smtpClient.DeliveryMethod = SmtpDeliveryMethod.Network;

                var message = new MailMessage(_emailFrom, _emailTo)
                {
                    Subject = $"[{severity.ToString().ToUpperInvariant()}] {subject}",
                    Body = $"Timestamp: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC\n\n{body}",
                    IsBodyHtml = false
                };

                await smtpClient.SendMailAsync(message);
                _logger.LogInformation("[ALERT] Email sent successfully - Subject: {Subject}", subject);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[ALERT] Failed to send email - Subject: {Subject}", subject);
                throw;
            }
        }

        public async Task SendSlackAsync(string message, AlertSeverity severity = AlertSeverity.Info)
        {
            if (string.IsNullOrEmpty(_slackWebhook))
            {
                _logger.LogWarning("[ALERT] Slack webhook not configured");
                return;
            }

            try
            {
                var emoji = GetSeverityEmoji(severity);
                var color = GetSeverityColor(severity);
                
                var payload = new
                {
                    text = $"{emoji} *Trading Bot Alert*",
                    attachments = new[]
                    {
                        new
                        {
                            color = color,
                            fields = new[]
                            {
                                new { title = "Severity", value = severity.ToString(), @short = true },
                                new { title = "Timestamp", value = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss UTC"), @short = true },
                                new { title = "Message", value = message, @short = false }
                            },
                            footer = "Trading Bot Alert System",
                            ts = DateTimeOffset.UtcNow.ToUnixTimeSeconds()
                        }
                    }
                };

                var json = JsonSerializer.Serialize(payload);
                var content = new StringContent(json, Encoding.UTF8, "application/json");
                
                var response = await _httpClient.PostAsync(_slackWebhook, content);
                response.EnsureSuccessStatusCode();
                
                _logger.LogInformation("[ALERT] Slack message sent successfully - Severity: {Severity}", severity);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[ALERT] Failed to send Slack message - Message: {Message}", message);
                throw;
            }
        }

        public async Task SendCriticalAlertAsync(string title, string details)
        {
            var subject = $"CRITICAL: {title}";
            var body = $"CRITICAL SYSTEM ALERT\n\nTitle: {title}\nDetails: {details}\n\nImmediate attention required!";
            
            // Send both email and Slack for critical alerts
            var emailTask = SendEmailAsync(subject, body, AlertSeverity.Critical);
            var slackTask = SendSlackAsync($"🚨 *CRITICAL ALERT*: {title}\n\n{details}", AlertSeverity.Critical);
            
            await Task.WhenAll(emailTask, slackTask);
        }

        public async Task SendModelHealthAlertAsync(string modelName, string healthIssue, object? metrics = null)
        {
            var metricsText = metrics != null ? JsonSerializer.Serialize(metrics, new JsonSerializerOptions { WriteIndented = true }) : "N/A";
            var subject = $"Model Health Alert: {modelName}";
            var body = $"Model: {modelName}\nIssue: {healthIssue}\nMetrics:\n{metricsText}";
            
            await SendEmailAsync(subject, body, AlertSeverity.Warning);
            await SendSlackAsync($"⚠️ **Model Health Alert**\nModel: `{modelName}`\nIssue: {healthIssue}", AlertSeverity.Warning);
        }

        public async Task SendLatencyAlertAsync(string component, double latencyMs, double thresholdMs)
        {
            var subject = $"Latency Alert: {component}";
            var body = $"Component: {component}\nLatency: {latencyMs:F2}ms\nThreshold: {thresholdMs:F2}ms\nExcess: {latencyMs - thresholdMs:F2}ms";
            
            await SendEmailAsync(subject, body, AlertSeverity.Warning);
            await SendSlackAsync($"🐌 **Latency Alert**\nComponent: `{component}`\nLatency: {latencyMs:F2}ms (threshold: {thresholdMs:F2}ms)", AlertSeverity.Warning);
        }

        public async Task SendDeploymentAlertAsync(string deploymentEvent, string modelName, bool isSuccess)
        {
            var severity = isSuccess ? AlertSeverity.Info : AlertSeverity.Error;
            var status = isSuccess ? "SUCCESS" : "FAILED";
            var emoji = isSuccess ? "✅" : "❌";
            
            var subject = $"Deployment {status}: {deploymentEvent}";
            var body = $"Event: {deploymentEvent}\nModel: {modelName}\nStatus: {status}";
            
            await SendEmailAsync(subject, body, severity);
            await SendSlackAsync($"{emoji} **Deployment Alert**\nEvent: {deploymentEvent}\nModel: `{modelName}`\nStatus: {status}", severity);
        }

        private static string GetSeverityEmoji(AlertSeverity severity) => severity switch
        {
            AlertSeverity.Critical => "🚨",
            AlertSeverity.Error => "❌",
            AlertSeverity.Warning => "⚠️",
            AlertSeverity.Info => "ℹ️",
            _ => "📢"
        };

        private static string GetSeverityColor(AlertSeverity severity) => severity switch
        {
            AlertSeverity.Critical => "#FF0000", // Red
            AlertSeverity.Error => "#FF6600",    // Orange-Red
            AlertSeverity.Warning => "#FFCC00",  // Yellow
            AlertSeverity.Info => "#0099CC",     // Blue
            _ => "#CCCCCC"                       // Gray
        };
    }

    public interface IAlertService
    {
        Task SendEmailAsync(string subject, string body, AlertSeverity severity = AlertSeverity.Info);
        Task SendSlackAsync(string message, AlertSeverity severity = AlertSeverity.Info);
        Task SendCriticalAlertAsync(string title, string details);
        Task SendModelHealthAlertAsync(string modelName, string healthIssue, object? metrics = null);
        Task SendLatencyAlertAsync(string component, double latencyMs, double thresholdMs);
        Task SendDeploymentAlertAsync(string deploymentEvent, string modelName, bool isSuccess);
    }

    public enum AlertSeverity
    {
        Info,
        Warning,
        Error,
        Critical
    }
}