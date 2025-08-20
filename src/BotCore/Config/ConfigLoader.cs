using System.Text.Json;

namespace BotCore.Config;

public static class ConfigLoader
{
    static readonly JsonSerializerOptions _opts = new()
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true
    };

    public static TradingProfileConfig FromFile(string path)
    {
        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<TradingProfileConfig>(json, _opts)
               ?? throw new InvalidOperationException("Failed to parse config.");
    }
}
