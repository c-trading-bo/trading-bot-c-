using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

public sealed class TopstepAuthAgent
{
    private readonly HttpClient _http;
    private static readonly JsonSerializerOptions JsonOpts = new(JsonSerializerDefaults.Web);

    public TopstepAuthAgent(HttpClient http)
    {
        _http = http;
        _http.BaseAddress ??= new Uri("https://api.topstepx.com");
        _http.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
    }

    public async Task<string> GetJwtAsync(string username, string apiKey, CancellationToken ct)
    {
        // IMPORTANT: /api/Auth/loginKey (exact path)
        var req = new HttpRequestMessage(HttpMethod.Post, "/api/Auth/loginKey")
        {
            // Use property names exactly as docs show: userName, apiKey
            Content = new StringContent(
                JsonSerializer.Serialize(new { userName = username, apiKey }),
                Encoding.UTF8, "application/json")
        };

        using var resp = await _http.SendAsync(req, ct);
        if (!resp.IsSuccessStatusCode)
        {
            var body = await resp.Content.ReadAsStringAsync(ct);
            throw new HttpRequestException($"Auth {(int)resp.StatusCode} {resp.StatusCode}: {body}", null, resp.StatusCode);
        }

        var responseBody = await resp.Content.ReadAsStringAsync(ct);
        using var doc = JsonDocument.Parse(responseBody);
        
        // Check if authentication was successful
        if (doc.RootElement.TryGetProperty("success", out var successProp) && 
            successProp.GetBoolean() == false)
        {
            var errorCode = doc.RootElement.TryGetProperty("errorCode", out var errCodeProp) ? errCodeProp.GetInt32() : 0;
            var errorMessage = doc.RootElement.TryGetProperty("errorMessage", out var errMsgProp) ? errMsgProp.GetString() : "Unknown error";
            throw new HttpRequestException($"TopstepX authentication failed - Error Code: {errorCode}, Message: {errorMessage ?? "No message"}");
        }
        
        return doc.RootElement.GetProperty("token").GetString()!;
    }

    public async Task<string?> ValidateAsync(CancellationToken ct)
    {
        var req = new HttpRequestMessage(HttpMethod.Post, "/api/Auth/validate");
        using var resp = await _http.SendAsync(req, ct);
        if (!resp.IsSuccessStatusCode) return null;

        using var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync(ct));
        if (doc.RootElement.TryGetProperty("newToken", out var nt)) return nt.GetString();
        return null;
    }
}

// Simple HttpRequestMessage.Clone() so we can resend the content on retries:
public static class HttpRequestMessageExtensions
{
    public static HttpRequestMessage Clone(this HttpRequestMessage req)
    {
        var clone = new HttpRequestMessage(req.Method, req.RequestUri);
        // Copy headers
        foreach (var h in req.Headers)
            clone.Headers.TryAddWithoutValidation(h.Key, h.Value);
        // Copy content
        if (req.Content != null)
        {
            var contentBytesTask = req.Content.ReadAsByteArrayAsync();
            contentBytesTask.Wait();
            var newContent = new ByteArrayContent(contentBytesTask.Result);
            foreach (var h in req.Content.Headers)
                newContent.Headers.TryAddWithoutValidation(h.Key, h.Value);
            clone.Content = newContent;
        }
        return clone;
    }
}
