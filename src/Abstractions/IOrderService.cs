using System.Threading.Tasks;

namespace TradingBot.Abstractions
{
    /// <summary>
    /// Interface for order service to avoid circular dependencies
    /// </summary>
    public interface IOrderService
    {
        Task<bool> IsHealthyAsync();
        Task<string> GetStatusAsync();
    }
}