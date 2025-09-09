using System;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using Moq;
using Moq.Protected;
using Xunit;
using TradingBot.IntelligenceAgent;

namespace TradingBot.Tests.Integration;

public class VerifierIntegrationTests
{
    [Fact]
    public async Task VerifyTodayAsync_SuccessfulResponse_ShouldReturnValidResult()
    {
        // Arrange
        var mockOrdersResponse = new
        {
            orders = new[]
            {
                new { status = "Filled", orderId = "order1" },
                new { status = "Placed", orderId = "order2" },
                new { status = "Cancelled", orderId = "order3" }
            }
        };

        var mockTradesResponse = new
        {
            trades = new[]
            {
                new { tradeId = "trade1", status = "Executed" },
                new { tradeId = "trade2", status = "Executed" }
            }
        };

        var mockHandler = new Mock<HttpMessageHandler>();
        
        // Mock orders endpoint
        mockHandler.Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.Is<HttpRequestMessage>(req => req.RequestUri!.ToString().Contains("/api/Order/search")),
                ItExpr.IsAny<CancellationToken>())
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.OK,
                Content = new StringContent(JsonSerializer.Serialize(mockOrdersResponse), Encoding.UTF8, "application/json")
            });

        // Mock trades endpoint  
        mockHandler.Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.Is<HttpRequestMessage>(req => req.RequestUri!.ToString().Contains("/api/Trade/search")),
                ItExpr.IsAny<CancellationToken>())
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.OK,
                Content = new StringContent(JsonSerializer.Serialize(mockTradesResponse), Encoding.UTF8, "application/json")
            });

        var httpClient = new HttpClient(mockHandler.Object)
        {
            BaseAddress = new Uri("https://api.topstepx.com")
        };

        var verifier = new Verifier(httpClient, NullLogger<Verifier>.Instance);

        // Act
        var result = await verifier.VerifyTodayAsync();

        // Assert
        Assert.True(result.Success);
        Assert.Equal(DateTime.UtcNow.Date, result.Date);
        Assert.Equal(1, result.OrdersByStatus["Filled"]);
        Assert.Equal(1, result.OrdersByStatus["Placed"]);
        Assert.Equal(1, result.OrdersByStatus["Cancelled"]);
        Assert.Equal(0, result.OrdersByStatus["PartiallyFilled"]);
        Assert.Equal(0, result.OrdersByStatus["Rejected"]);
        Assert.Equal(2, result.TradesByStatus["Executed"]);
    }

    [Fact]
    public async Task VerifyTodayAsync_OrdersEndpointFails_ShouldHandleGracefully()
    {
        // Arrange
        var mockTradesResponse = new
        {
            trades = new[]
            {
                new { tradeId = "trade1" }
            }
        };

        var mockHandler = new Mock<HttpMessageHandler>();
        
        // Mock orders endpoint - return error
        mockHandler.Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.Is<HttpRequestMessage>(req => req.RequestUri!.ToString().Contains("/api/Order/search")),
                ItExpr.IsAny<CancellationToken>())
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.InternalServerError,
                Content = new StringContent("Server Error")
            });

        // Mock trades endpoint - return success
        mockHandler.Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.Is<HttpRequestMessage>(req => req.RequestUri!.ToString().Contains("/api/Trade/search")),
                ItExpr.IsAny<CancellationToken>())
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.OK,
                Content = new StringContent(JsonSerializer.Serialize(mockTradesResponse), Encoding.UTF8, "application/json")
            });

        var httpClient = new HttpClient(mockHandler.Object)
        {
            BaseAddress = new Uri("https://api.topstepx.com")
        };

        var verifier = new Verifier(httpClient, NullLogger<Verifier>.Instance);

        // Act
        var result = await verifier.VerifyTodayAsync();

        // Assert
        Assert.True(result.Success);
        Assert.Equal(0, result.OrdersByStatus.Values.Sum()); // No orders due to API error
        Assert.Equal(1, result.TradesByStatus["Executed"]); // Trades still processed
    }

    [Fact]
    public async Task VerifyTodayAsync_MalformedJson_ShouldHandleGracefully()
    {
        // Arrange
        var mockHandler = new Mock<HttpMessageHandler>();
        
        // Mock both endpoints with malformed JSON
        mockHandler.Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.IsAny<HttpRequestMessage>(),
                ItExpr.IsAny<CancellationToken>())
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.OK,
                Content = new StringContent("{ invalid json", Encoding.UTF8, "application/json")
            });

        var httpClient = new HttpClient(mockHandler.Object)
        {
            BaseAddress = new Uri("https://api.topstepx.com")
        };

        var verifier = new Verifier(httpClient, NullLogger<Verifier>.Instance);

        // Act
        var result = await verifier.VerifyTodayAsync();

        // Assert
        Assert.True(result.Success);
        Assert.Equal(0, result.OrdersByStatus.Values.Sum());
        Assert.Equal(0, result.TradesByStatus.Values.Sum());
    }

    [Fact]
    public async Task VerifyTodayAsync_HttpException_ShouldReturnFailure()
    {
        // Arrange
        var mockHandler = new Mock<HttpMessageHandler>();
        
        mockHandler.Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.IsAny<HttpRequestMessage>(),
                ItExpr.IsAny<CancellationToken>())
            .ThrowsAsync(new HttpRequestException("Network error"));

        var httpClient = new HttpClient(mockHandler.Object)
        {
            BaseAddress = new Uri("https://api.topstepx.com")
        };

        var verifier = new Verifier(httpClient, NullLogger<Verifier>.Instance);

        // Act
        var result = await verifier.VerifyTodayAsync();

        // Assert
        Assert.False(result.Success);
        Assert.NotNull(result.ErrorMessage);
        Assert.Contains("Network error", result.ErrorMessage);
    }

    [Fact]
    public async Task VerifyTodayAsync_CancellationRequested_ShouldThrow()
    {
        // Arrange
        var mockHandler = new Mock<HttpMessageHandler>();
        var httpClient = new HttpClient(mockHandler.Object)
        {
            BaseAddress = new Uri("https://api.topstepx.com")
        };

        var verifier = new Verifier(httpClient, NullLogger<Verifier>.Instance);
        var cancellationToken = new CancellationToken(true); // Already cancelled

        // Act & Assert
        await Assert.ThrowsAsync<OperationCanceledException>(
            () => verifier.VerifyTodayAsync(cancellationToken));
    }

    [Fact]
    public async Task VerifyTodayAsync_EmptyResponse_ShouldReturnZeroCounts()
    {
        // Arrange
        var emptyOrdersResponse = new { orders = Array.Empty<object>() };
        var emptyTradesResponse = new { trades = Array.Empty<object>() };

        var mockHandler = new Mock<HttpMessageHandler>();
        
        mockHandler.Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.Is<HttpRequestMessage>(req => req.RequestUri!.ToString().Contains("/api/Order/search")),
                ItExpr.IsAny<CancellationToken>())
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.OK,
                Content = new StringContent(JsonSerializer.Serialize(emptyOrdersResponse), Encoding.UTF8, "application/json")
            });

        mockHandler.Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.Is<HttpRequestMessage>(req => req.RequestUri!.ToString().Contains("/api/Trade/search")),
                ItExpr.IsAny<CancellationToken>())
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.OK,
                Content = new StringContent(JsonSerializer.Serialize(emptyTradesResponse), Encoding.UTF8, "application/json")
            });

        var httpClient = new HttpClient(mockHandler.Object)
        {
            BaseAddress = new Uri("https://api.topstepx.com")
        };

        var verifier = new Verifier(httpClient, NullLogger<Verifier>.Instance);

        // Act
        var result = await verifier.VerifyTodayAsync();

        // Assert
        Assert.True(result.Success);
        Assert.Equal(0, result.OrdersByStatus.Values.Sum());
        Assert.Equal(0, result.TradesByStatus.Values.Sum());
    }
}