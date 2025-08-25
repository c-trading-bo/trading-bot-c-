#nullable enable
using System;
using System.IO;
using System.Text.Json;

namespace BotCore.Infra
{
    public static class Persistence
    {
        static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "state");

        public static void Save<T>(string name, T obj)
        {
            try
            {
                Directory.CreateDirectory(Dir);
                var path = Path.Combine(Dir, $"{name}.json");
                var json = JsonSerializer.Serialize(obj, new JsonSerializerOptions { WriteIndented = false });
                File.WriteAllText(path, json);
            }
            catch { }
        }

        public static T? Load<T>(string name)
        {
            try
            {
                var path = Path.Combine(Dir, $"{name}.json");
                if (!File.Exists(path)) return default;
                var json = File.ReadAllText(path);
                return JsonSerializer.Deserialize<T>(json);
            }
            catch { return default; }
        }
    }
}
