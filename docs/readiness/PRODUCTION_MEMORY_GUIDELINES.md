# Production Memory Management Guidelines

## Issues with Current Implementation
- Multiple forced GC.Collect() calls are anti-patterns in production
- Aggressive garbage collection causes performance spikes
- LOH compaction can cause application pauses

## Recommended Approach
1. **Memory Pressure Monitoring**: Use GC notification APIs instead of forced collection
2. **Proper Disposal**: Implement IDisposable pattern correctly
3. **Weak References**: Use weak references for cacheable objects
4. **Memory Budgets**: Set memory limits and monitor usage
5. **Lazy Loading**: Load models on-demand instead of preloading

## Production Memory Settings
```csharp
// Remove these anti-patterns:
GC.Collect(2, GCCollectionMode.Forced); // NEVER do this in production

// Replace with:
if (GC.GetTotalMemory(false) > memoryThreshold)
{
    // Cleanup specific objects instead of forcing GC
    CleanupUnusedModels();
}
```

## Environment Configuration
```
DOTNET_GCServer=1
DOTNET_GCConcurrent=1
DOTNET_GCRetainVM=1
DOTNET_GCLOHThreshold=85000
```