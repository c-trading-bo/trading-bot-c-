using System;
using System.Threading.Tasks;

namespace TradingBot.Abstractions
{
    /// <summary>
    /// Event args for status changes
    /// </summary>
    public class StatusChangedEventArgs : EventArgs
    {
        public string Status { get; }
        public StatusChangedEventArgs(string status) => Status = status;
    }

    /// <summary>
    /// Interface for TopstepX adapter service to avoid circular dependencies
    /// </summary>
    public interface ITopstepXAdapterService
    {
        Task<bool> IsConnectedAsync();
        Task<string> GetAccountStatusAsync();
        event EventHandler<StatusChangedEventArgs>? StatusChanged;
    }
}