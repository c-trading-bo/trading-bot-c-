using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace BotCore.StrategyDsl;

/// <summary>
/// Simple DSL loader for strategy YAML files
/// Matches the problem statement specification exactly
/// </summary>
public static class SimpleDslLoader
{
    /// <summary>
    /// Load all YAML strategy files from the specified folder
    /// Returns deserialized DslStrategy objects ready for knowledge graph evaluation
    /// </summary>
    /// <param name="folder">Path to folder containing YAML strategy files</param>
    /// <returns>List of loaded DSL strategies</returns>
    public static IReadOnlyList<DslStrategy> LoadAll(string folder)
    {
        if (!Directory.Exists(folder))
        {
            throw new DirectoryNotFoundException($"Strategy folder not found: {folder}");
        }

        var deserializer = new DeserializerBuilder()
            .WithNamingConvention(CamelCaseNamingConvention.Instance)
            .IgnoreUnmatchedProperties()
            .Build();

        var strategies = new List<DslStrategy>();
        var yamlFiles = Directory.EnumerateFiles(folder, "*.yaml")
                                .Concat(Directory.EnumerateFiles(folder, "*.yml"));

        foreach (var file in yamlFiles)
        {
            try
            {
                var yamlContent = File.ReadAllText(file);
                var strategy = deserializer.Deserialize<YamlStrategy>(yamlContent);
                
                if (strategy != null && !string.IsNullOrEmpty(strategy.Name))
                {
                    var dslStrategy = ConvertToDslStrategy(strategy);
                    strategies.Add(dslStrategy);
                }
            }
            catch (Exception ex)
            {
                // Skip strategies that don't match the simple DSL format
                // Complex strategies (like S7) may have their own loaders
                // Only show warning if not in Lab Mode (dashboard-only view)
                var labMode = Environment.GetEnvironmentVariable("LAB_MODE");
                if (labMode != "1" && labMode?.ToLowerInvariant() != "true")
                {
                    Console.WriteLine($"⚠️ [DSL-LOADER] Skipping {Path.GetFileName(file)}: {ex.Message}");
                }
            }
        }

        return strategies;
    }

    /// <summary>
    /// Convert YAML strategy format to internal DSL strategy format
    /// </summary>
    private static DslStrategy ConvertToDslStrategy(YamlStrategy yaml)
    {
        return new DslStrategy
        {
            Name = yaml.Name,
            Label = yaml.Label,
            Family = yaml.Family,
            Bias = yaml.Bias,
            TelemetryTags = yaml.TelemetryTags ?? new List<string>(),
            When = new DslWhen
            {
                Regime = yaml.When?.Regime ?? new List<string>(),
                Micro = yaml.When?.Micro ?? new List<string>()
            },
            Contra = yaml.Contra,
            Confluence = yaml.Confluence ?? new List<string>(),
            Playbook = yaml.Playbook != null ? new DslPlaybook 
            { 
                Name = yaml.Playbook.Entry + "; " + yaml.Playbook.Bracket,
                Description = $"Entry: {yaml.Playbook.Entry}, Bracket: {yaml.Playbook.Bracket}"
            } : null
        };
    }
}

/// <summary>
/// YAML strategy structure matching the problem statement format
/// </summary>
public class YamlStrategy
{
    public string Name { get; set; } = string.Empty;
    public string Label { get; set; } = string.Empty;
    public string Family { get; set; } = string.Empty;
    public string Bias { get; set; } = "both";
    public YamlWhen? When { get; set; }
    public List<string>? Contra { get; set; }
    public List<string>? Confluence { get; set; }
    public YamlPlaybook? Playbook { get; set; }
    public List<string>? TelemetryTags { get; set; }
}

/// <summary>
/// YAML when conditions structure
/// </summary>
public class YamlWhen
{
    public List<string>? Regime { get; set; }
    public List<string>? Micro { get; set; }
}

/// <summary>
/// YAML playbook structure
/// </summary>
public class YamlPlaybook
{
    public string Entry { get; set; } = string.Empty;
    public string Bracket { get; set; } = string.Empty;
}