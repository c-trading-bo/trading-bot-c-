using System;
using System.Collections.Generic;
using System.Text.Json;

namespace BotCore.Models
{
    /// <summary>
    /// Parameter store override records for strategy configurations
    /// </summary>
    public sealed record S2Override(string SymbolRoot, DateTime ExpiresUtc, Dictionary<string, JsonElement> Extra);
    
    public sealed record S3Override(string SymbolRoot, DateTime ExpiresUtc, string JsonConfig);
    
    public sealed record S6Override(string SymbolRoot, DateTime ExpiresUtc, Dictionary<string, JsonElement> Extra);
    
    public sealed record S11Override(string SymbolRoot, DateTime ExpiresUtc, Dictionary<string, JsonElement> Extra);
}