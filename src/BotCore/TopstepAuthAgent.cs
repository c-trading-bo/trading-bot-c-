using System;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore
{
    public class TopstepAuthAgent
    {
        public string? Token { get; private set; }
        public string? Username { get; }
        public string? ApiKey { get; }
        public string EnvFilePath { get; }

        public TopstepAuthAgent(string username, string apiKey, string envFilePath = ".env.local")
        {
            Username = username;
            ApiKey = apiKey;
            EnvFilePath = envFilePath;
        }

        public async Task<bool> LoginAsync(CancellationToken ct = default)
        {
            using var client = new HttpClient { BaseAddress = new Uri("https://api.topstepx.com") };
            var body = new { userName = Username, apiKey = ApiKey };
            var resp = await client.PostAsJsonAsync("/api/Auth/loginKey", body, ct);
            if (!resp.IsSuccessStatusCode)
            {
                Console.WriteLine($"Failed to fetch JWT: {resp.StatusCode}");
                return false;
            }
            var respText = await resp.Content.ReadAsStringAsync(ct);
            try
            {
                var json = JsonDocument.Parse(respText);
                var root = json.RootElement;
                bool success = root.TryGetProperty("success", out var successProp) && successProp.GetBoolean();
                int errorCode = root.TryGetProperty("errorCode", out var errorCodeProp) ? errorCodeProp.GetInt32() : -1;
                string? errorMessage = root.TryGetProperty("errorMessage", out var errorMsgProp) ? errorMsgProp.GetString() : null;
                if (!success || errorCode != 0)
                {
                    Console.WriteLine($"Login failed: errorCode={errorCode}, errorMessage={errorMessage}");
                    return false;
                }
                if (root.TryGetProperty("token", out var jwtProp))
                {
                    Token = jwtProp.GetString();
                    Environment.SetEnvironmentVariable("TOPSTEPX_JWT", Token);
                    UpdateEnvFile(Token);
                    Console.WriteLine("Fetched new JWT and set TOPSTEPX_JWT.");
                    return true;
                }
                else { Console.WriteLine("JWT not found in response JSON."); return false; }
            }
            catch (Exception ex) { Console.WriteLine($"Error parsing JWT response: {ex.Message}"); return false; }
        }

        private void UpdateEnvFile(string token)
        {
            var lines = System.IO.File.Exists(EnvFilePath) ? System.IO.File.ReadAllLines(EnvFilePath).ToList() : new System.Collections.Generic.List<string>();
            bool found = false;
            for (int i = 0; i < lines.Count; i++)
            {
                if (lines[i].StartsWith("TOPSTEPX_JWT=")) { lines[i] = $"TOPSTEPX_JWT={token}"; found = true; break; }
            }
            if (!found) lines.Add($"TOPSTEPX_JWT={token}");
            System.IO.File.WriteAllLines(EnvFilePath, lines);
            Console.WriteLine("Updated .env.local with new JWT token.");
        }
    }
}
