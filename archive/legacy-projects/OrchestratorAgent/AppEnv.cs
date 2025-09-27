using System;

namespace OrchestratorAgent
{
    internal static class AppEnv
    {
        public static string Get(string key, string? def = null)
            => Environment.GetEnvironmentVariable(key) ?? def ?? throw new($"Missing env {key}");

        public static int Int(string key, int def)
            => int.TryParse(Environment.GetEnvironmentVariable(key), out var v) ? v : def;

        public static bool Flag(string key, bool defaultTrue)
        {
            var raw = Environment.GetEnvironmentVariable(key);
            if (string.IsNullOrWhiteSpace(raw)) return defaultTrue;
            raw = raw.Trim();
            return raw.Equals("1", StringComparison.OrdinalIgnoreCase)
                || raw.Equals("true", StringComparison.OrdinalIgnoreCase)
                || raw.Equals("yes", StringComparison.OrdinalIgnoreCase);
        }

        public static void Set(string key, string value) => Environment.SetEnvironmentVariable(key, value);
    }
}
