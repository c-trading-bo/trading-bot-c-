namespace BotCore.ML;

/// <summary>
/// Interface for ML memory management services
/// </summary>
public interface IMLMemoryManager : IDisposable
{
    /// <summary>
    /// Initialize memory management services
    /// </summary>
    Task InitializeMemoryManagementAsync();
    
    /// <summary>
    /// Load and manage ML model with memory tracking
    /// </summary>
    Task<T?> LoadModelAsync<T>(string modelPath, string version) where T : class;
    
    /// <summary>
    /// Get current memory usage statistics
    /// </summary>
    MLMemoryManager.MemorySnapshot GetMemorySnapshot();
}