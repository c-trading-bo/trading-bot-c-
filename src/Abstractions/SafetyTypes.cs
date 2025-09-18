using System;
using System.Collections.Generic;

namespace TopstepX.Bot.Abstractions;

// Emergency Stop Event Arguments
public class EmergencyStopEventArgs : EventArgs
{
    public string Reason { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
}

// Risk Management Event Arguments  
public class RiskViolationEventArgs : EventArgs
{
    public string Symbol { get; set; } = string.Empty;
    public string ViolationType { get; set; } = string.Empty;
    public List<string> Violations { get; } = new();
    public object? Position { get; set; } // Generic position object
    public DateTime Timestamp { get; set; }
}

// API Response Types
public class ApiOrderResponse
{
    public bool IsSuccess { get; set; }
    public string? OrderId { get; set; }
    public string? ErrorMessage { get; set; }
    
    public static ApiOrderResponse Success(string orderId) => new() { IsSuccess = true, OrderId = orderId };
    public static ApiOrderResponse Failed(string error) => new() { IsSuccess = false, ErrorMessage = error };
}

public class ApiOrderDetails
{
    public string OrderId { get; set; } = string.Empty;
    public string Status { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public int Quantity { get; set; }
    public decimal Price { get; set; }
}

// Account Summary
public class AccountSummary
{
    public decimal AccountBalance { get; set; }
    public decimal TotalDailyPnL { get; set; }
    public decimal TotalUnrealizedPnL { get; set; }
    public decimal TotalRealizedPnL { get; set; }
    public decimal TotalMarketValue { get; set; }
    public int OpenPositions { get; set; }
    public int PendingOrders { get; set; }
    public DateTime LastUpdate { get; set; }
}