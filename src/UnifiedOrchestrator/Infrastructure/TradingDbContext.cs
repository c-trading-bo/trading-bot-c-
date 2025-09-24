using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Threading.Tasks;
using TradingBot.UnifiedOrchestrator.Services;

namespace TradingBot.UnifiedOrchestrator.Infrastructure;

/// <summary>
/// Production-grade database context for trading operations
/// Implements ACID transactions, proper schema management, and data integrity
/// </summary>
internal class TradingDbContext : DbContext, ITradingDbContext
{
    private readonly ILogger<TradingDbContext> _logger;

    public TradingDbContext(DbContextOptions<TradingDbContext> options, ILogger<TradingDbContext> logger) 
        : base(options)
    {
        _logger = logger;
    }

    public DbSet<TradeEntity> Trades { get; set; } = null!;
    public DbSet<PositionEntity> Positions { get; set; } = null!;
    public DbSet<OrderEntity> Orders { get; set; } = null!;
    public DbSet<AccountStateEntity> AccountStates { get; set; } = null!;
    public DbSet<RiskEventEntity> RiskEvents { get; set; } = null!;
    public DbSet<PerformanceMetricEntity> PerformanceMetrics { get; set; } = null!;

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        base.OnModelCreating(modelBuilder);

        // Configure Trade entity
        modelBuilder.Entity<TradeEntity>(entity =>
        {
            entity.HasKey(e => e.Id);
            entity.Property(e => e.Symbol).IsRequired().HasMaxLength(20);
            entity.Property(e => e.Side).IsRequired().HasMaxLength(10);
            entity.Property(e => e.Price).HasPrecision(18, 8);
            entity.Property(e => e.Quantity).HasPrecision(18, 8);
            entity.Property(e => e.Commission).HasPrecision(18, 8);
            entity.HasIndex(e => new { e.Symbol, e.Timestamp });
            entity.HasIndex(e => e.OrderId);
            entity.HasIndex(e => e.Timestamp);
        });

        // Configure Position entity
        modelBuilder.Entity<PositionEntity>(entity =>
        {
            entity.HasKey(e => e.Id);
            entity.Property(e => e.Symbol).IsRequired().HasMaxLength(20);
            entity.Property(e => e.Quantity).HasPrecision(18, 8);
            entity.Property(e => e.AveragePrice).HasPrecision(18, 8);
            entity.Property(e => e.UnrealizedPnL).HasPrecision(18, 8);
            entity.HasIndex(e => new { e.AccountId, e.Symbol }).IsUnique();
            entity.HasIndex(e => e.LastUpdated);
        });

        // Configure Order entity
        modelBuilder.Entity<OrderEntity>(entity =>
        {
            entity.HasKey(e => e.Id);
            entity.Property(e => e.Symbol).IsRequired().HasMaxLength(20);
            entity.Property(e => e.Side).IsRequired().HasMaxLength(10);
            entity.Property(e => e.OrderType).IsRequired().HasMaxLength(20);
            entity.Property(e => e.Price).HasPrecision(18, 8);
            entity.Property(e => e.Quantity).HasPrecision(18, 8);
            entity.Property(e => e.FilledQuantity).HasPrecision(18, 8);
            entity.Property(e => e.AveragePrice).HasPrecision(18, 8);
            entity.HasIndex(e => e.OrderId).IsUnique();
            entity.HasIndex(e => new { e.Status, e.Timestamp });
        });

        // Configure AccountState entity
        modelBuilder.Entity<AccountStateEntity>(entity =>
        {
            entity.HasKey(e => e.Id);
            entity.Property(e => e.Balance).HasPrecision(18, 8);
            entity.Property(e => e.Equity).HasPrecision(18, 8);
            entity.Property(e => e.MarginUsed).HasPrecision(18, 8);
            entity.Property(e => e.MarginAvailable).HasPrecision(18, 8);
            entity.Property(e => e.DayPnL).HasPrecision(18, 8);
            entity.Property(e => e.TotalPnL).HasPrecision(18, 8);
            entity.HasIndex(e => new { e.AccountId, e.Timestamp });
        });

        // Configure RiskEvent entity
        modelBuilder.Entity<RiskEventEntity>(entity =>
        {
            entity.HasKey(e => e.Id);
            entity.Property(e => e.EventType).IsRequired().HasMaxLength(50);
            entity.Property(e => e.Severity).IsRequired().HasMaxLength(20);
            entity.Property(e => e.RiskValue).HasPrecision(18, 8);
            entity.Property(e => e.ThresholdValue).HasPrecision(18, 8);
            entity.HasIndex(e => new { e.EventType, e.Timestamp });
            entity.HasIndex(e => e.Severity);
        });

        // Configure PerformanceMetric entity
        modelBuilder.Entity<PerformanceMetricEntity>(entity =>
        {
            entity.HasKey(e => e.Id);
            entity.Property(e => e.MetricName).IsRequired().HasMaxLength(100);
            entity.Property(e => e.MetricValue).HasPrecision(18, 8);
            entity.HasIndex(e => new { e.MetricName, e.Timestamp });
        });
    }

    public async Task TestConnectionAsync()
    {
        try
        {
            _logger.LogInformation("[DATABASE] Testing database connection...");
            await Database.EnsureCreatedAsync().ConfigureAwait(false);
            
            // Test a simple query
            var tradeCount = await Trades.CountAsync().ConfigureAwait(false);
            _logger.LogInformation("[DATABASE] Connection successful. Trade count: {TradeCount}", tradeCount);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DATABASE] Connection test failed");
            throw;
        }
    }

    public async Task SaveTradeAsync(TradeRecord trade)
    {
        using var transaction = await Database.BeginTransactionAsync().ConfigureAwait(false);
        try
        {
            var entity = new TradeEntity
            {
                Id = trade.Id,
                Symbol = trade.Symbol,
                Side = trade.Side,
                Quantity = trade.Quantity,
                Price = trade.Price,
                Timestamp = trade.Timestamp,
                OrderId = trade.OrderId,
                Commission = trade.Commission,
                Status = trade.Status
            };

            Trades.Add(entity);
            await SaveChangesAsync().ConfigureAwait(false);
            await transaction.CommitAsync().ConfigureAwait(false);

            _logger.LogInformation("[DATABASE] Trade saved: {TradeId} {Symbol} {Side} {Quantity}@{Price}", 
                trade.Id, trade.Symbol, trade.Side, trade.Quantity, trade.Price);
        }
        catch (Exception ex)
        {
            await transaction.RollbackAsync().ConfigureAwait(false);
            _logger.LogError(ex, "[DATABASE] Failed to save trade: {TradeId}", trade.Id);
            throw;
        }
    }

    public async Task SavePositionAsync(PositionRecord position)
    {
        using var transaction = await Database.BeginTransactionAsync().ConfigureAwait(false);
        try
        {
            var existing = await Positions
                .FirstOrDefaultAsync(p => p.AccountId == position.AccountId && p.Symbol == position.Symbol).ConfigureAwait(false);

            if (existing != null)
            {
                existing.Quantity = position.Quantity;
                existing.AveragePrice = position.AveragePrice;
                existing.UnrealizedPnL = position.UnrealizedPnL;
                existing.LastUpdated = position.LastUpdated;
            }
            else
            {
                var entity = new PositionEntity
                {
                    Id = position.Id,
                    Symbol = position.Symbol,
                    Quantity = position.Quantity,
                    AveragePrice = position.AveragePrice,
                    UnrealizedPnL = position.UnrealizedPnL,
                    LastUpdated = position.LastUpdated,
                    AccountId = position.AccountId
                };
                Positions.Add(entity);
            }

            await SaveChangesAsync().ConfigureAwait(false);
            await transaction.CommitAsync().ConfigureAwait(false);

            _logger.LogInformation("[DATABASE] Position saved: {AccountId} {Symbol} Qty:{Quantity} Avg:{AveragePrice}", 
                position.AccountId, position.Symbol, position.Quantity, position.AveragePrice);
        }
        catch (Exception ex)
        {
            await transaction.RollbackAsync().ConfigureAwait(false);
            _logger.LogError(ex, "[DATABASE] Failed to save position: {AccountId} {Symbol}", position.AccountId, position.Symbol);
            throw;
        }
    }

    public async Task<List<TradeRecord>> GetTradeHistoryAsync(DateTime from, DateTime to)
    {
        try
        {
            var entities = await Trades
                .Where(t => t.Timestamp >= from && t.Timestamp <= to)
                .OrderByDescending(t => t.Timestamp)
                .ToListAsync().ConfigureAwait(false);

            return entities.Select(e => new TradeRecord
            {
                Id = e.Id,
                Symbol = e.Symbol,
                Side = e.Side,
                Quantity = e.Quantity,
                Price = e.Price,
                Timestamp = e.Timestamp,
                OrderId = e.OrderId,
                Commission = e.Commission,
                Status = e.Status
            }).ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DATABASE] Failed to get trade history from {From} to {To}", from, to);
            throw;
        }
    }

    public async Task SaveOrderAsync(OrderRecord order)
    {
        using var transaction = await Database.BeginTransactionAsync().ConfigureAwait(false);
        try
        {
            var existing = await Orders.FirstOrDefaultAsync(o => o.OrderId == order.OrderId).ConfigureAwait(false);

            if (existing != null)
            {
                existing.Status = order.Status;
                existing.FilledQuantity = order.FilledQuantity;
                existing.AveragePrice = order.AveragePrice;
                existing.LastUpdated = order.LastUpdated;
            }
            else
            {
                var entity = new OrderEntity
                {
                    Id = Guid.NewGuid().ToString(),
                    OrderId = order.OrderId,
                    Symbol = order.Symbol,
                    Side = order.Side,
                    OrderType = order.OrderType,
                    Quantity = order.Quantity,
                    Price = order.Price,
                    FilledQuantity = order.FilledQuantity,
                    AveragePrice = order.AveragePrice,
                    Status = order.Status,
                    Timestamp = order.Timestamp,
                    LastUpdated = order.LastUpdated
                };
                Orders.Add(entity);
            }

            await SaveChangesAsync().ConfigureAwait(false);
            await transaction.CommitAsync().ConfigureAwait(false);

            _logger.LogInformation("[DATABASE] Order saved: {OrderId} {Symbol} {Side} {Quantity}@{Price} Status:{Status}", 
                order.OrderId, order.Symbol, order.Side, order.Quantity, order.Price, order.Status);
        }
        catch (Exception ex)
        {
            await transaction.RollbackAsync().ConfigureAwait(false);
            _logger.LogError(ex, "[DATABASE] Failed to save order: {OrderId}", order.OrderId);
            throw;
        }
    }

    public async Task SaveRiskEventAsync(RiskEvent riskEvent)
    {
        try
        {
            var entity = new RiskEventEntity
            {
                Id = Guid.NewGuid().ToString(),
                EventType = riskEvent.EventType,
                Severity = riskEvent.Severity.ToString(),
                Description = riskEvent.Description,
                RiskValue = riskEvent.RiskValue,
                ThresholdValue = riskEvent.ThresholdValue,
                Symbol = riskEvent.Symbol,
                Timestamp = riskEvent.Timestamp
            };

            RiskEvents.Add(entity);
            await SaveChangesAsync().ConfigureAwait(false);

            _logger.LogInformation("[DATABASE] Risk event saved: {EventType} {Severity} {Symbol}", 
                riskEvent.EventType, riskEvent.Severity, riskEvent.Symbol);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DATABASE] Failed to save risk event: {EventType}", riskEvent.EventType);
            throw;
        }
    }
}

// Database entity classes
internal class TradeEntity
{
    public string Id { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public string Side { get; set; } = string.Empty;
    public decimal Quantity { get; set; }
    public decimal Price { get; set; }
    public DateTime Timestamp { get; set; }
    public string OrderId { get; set; } = string.Empty;
    public decimal Commission { get; set; }
    public string Status { get; set; } = string.Empty;
}

internal class PositionEntity
{
    public string Id { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public decimal Quantity { get; set; }
    public decimal AveragePrice { get; set; }
    public decimal UnrealizedPnL { get; set; }
    public DateTime LastUpdated { get; set; }
    public string AccountId { get; set; } = string.Empty;
}

internal class OrderEntity
{
    public string Id { get; set; } = string.Empty;
    public string OrderId { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public string Side { get; set; } = string.Empty;
    public string OrderType { get; set; } = string.Empty;
    public decimal Quantity { get; set; }
    public decimal Price { get; set; }
    public decimal FilledQuantity { get; set; }
    public decimal AveragePrice { get; set; }
    public string Status { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public DateTime LastUpdated { get; set; }
}

internal class AccountStateEntity
{
    public string Id { get; set; } = string.Empty;
    public string AccountId { get; set; } = string.Empty;
    public decimal Balance { get; set; }
    public decimal Equity { get; set; }
    public decimal MarginUsed { get; set; }
    public decimal MarginAvailable { get; set; }
    public decimal DayPnL { get; set; }
    public decimal TotalPnL { get; set; }
    public DateTime Timestamp { get; set; }
}

internal class RiskEventEntity
{
    public string Id { get; set; } = string.Empty;
    public string EventType { get; set; } = string.Empty;
    public string Severity { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public decimal RiskValue { get; set; }
    public decimal ThresholdValue { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
}

internal class PerformanceMetricEntity
{
    public string Id { get; set; } = string.Empty;
    public string MetricName { get; set; } = string.Empty;
    public decimal MetricValue { get; set; }
    public string Unit { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
}

// Supporting data classes
internal class OrderRecord
{
    public string OrderId { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public string Side { get; set; } = string.Empty;
    public string OrderType { get; set; } = string.Empty;
    public decimal Quantity { get; set; }
    public decimal Price { get; set; }
    public decimal FilledQuantity { get; set; }
    public decimal AveragePrice { get; set; }
    public string Status { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public DateTime LastUpdated { get; set; } = DateTime.UtcNow;
}

internal class RiskEvent
{
    public string EventType { get; set; } = string.Empty;
    public RiskSeverity Severity { get; set; }
    public string Description { get; set; } = string.Empty;
    public decimal RiskValue { get; set; }
    public decimal ThresholdValue { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

internal enum RiskSeverity
{
    Low,
    Medium,
    High,
    Critical
}

/// <summary>
/// Database service extensions for dependency injection
/// </summary>
internal static class DatabaseServiceExtensions
{
    public static IServiceCollection AddProductionDatabase(this IServiceCollection services, IConfiguration configuration)
    {
        var connectionString = configuration.GetConnectionString("TradingDatabase") 
            ?? "Data Source=trading.db;Cache=Shared;";

        services.AddDbContext<TradingDbContext>(options =>
        {
            options.UseSqlite(connectionString);
            options.EnableSensitiveDataLogging(false); // Disable in production
            options.EnableServiceProviderCaching();
            options.EnableDetailedErrors(false); // Disable in production
        });

        services.AddScoped<ITradingDbContext>(provider => provider.GetRequiredService<TradingDbContext>());

        return services;
    }
}