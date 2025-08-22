// Agent: OrderRouterAgent
// Role: Handles order placement, routing, and execution logic.
// Integration: Receives signals/orders from orchestrator and strategy agents.
using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using BotCore;

namespace OrderRouterAgent
{
	/// <summary>
	/// Handles order routing via ApiClient, including order placement and status logging.
	/// </summary>
	public sealed class OrderRouterAgent
	{
		private readonly ILogger<OrderRouterAgent> _log;
		private readonly ApiClient _api;
		private readonly int _accountId;

		public OrderRouterAgent(ILogger<OrderRouterAgent> log, ApiClient api, int accountId)
		{
			_log = log;
			_api = api;
			_accountId = accountId;
		}

		public async Task<string?> PlaceOrderAsync(object req, CancellationToken ct)
		{
			try
			{
				var orderId = await _api.PlaceOrderAsync(req, ct);
				_log.LogInformation($"Order placed: {orderId}");
				return orderId;
			}
			catch (Exception ex)
			{
				_log.LogWarning(ex, "Order placement failed");
				return null;
			}
		}

		public async Task PrintOrderStatusAsync(object searchBody, CancellationToken ct)
		{
			try
			{
				var orders = await _api.SearchOrdersAsync(searchBody, ct);
				_log.LogInformation($"Order search result: {orders}");
			}
			catch (Exception ex)
			{
				_log.LogWarning(ex, "Order search failed");
			}
		}
	}
}
