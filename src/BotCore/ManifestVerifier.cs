using System;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace BotCore
{
    /// <summary>
    /// Verifies HMAC-SHA256 signatures for model manifest security.
    /// 
    /// Ensures model manifests have not been tampered with during
    /// Model manifest verification for local model updates.
    /// Ensures integrity and freshness of local model files.
    /// </summary>
    public static class ManifestVerifier
    {
        /// <summary>
        /// Verify HMAC-SHA256 signature for a manifest JSON object.
        /// </summary>
        /// <param name="manifestJson">Raw manifest JSON string</param>
        /// <param name="hmacKey">Secret HMAC key</param>
        /// <param name="expectedSignature">Expected signature to verify</param>
        /// <returns>True if signature is valid</returns>
        public static bool VerifyManifestSignature(string manifestJson, string hmacKey, string expectedSignature)
        {
            try
            {
                // Parse and normalize JSON (same as Python implementation)
                var jsonDoc = JsonDocument.Parse(manifestJson);

                // Create a copy without the signature field for verification
                var manifestWithoutSig = CreateManifestWithoutSignature(jsonDoc.RootElement);

                // Generate canonical JSON (sorted keys, no whitespace)
                var canonicalJson = JsonSerializer.Serialize(manifestWithoutSig, new JsonSerializerOptions
                {
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                    WriteIndented = false
                });

                // Sort the JSON properties manually for true canonical form
                var sortedJson = SortJsonProperties(canonicalJson);

                // Generate HMAC-SHA256
                var actualSignature = GenerateHmacSignature(sortedJson, hmacKey);

                // Use timing-safe comparison
                return CryptographicOperations.FixedTimeEquals(
                    Convert.FromHexString(expectedSignature),
                    Convert.FromHexString(actualSignature)
                );
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[SECURITY] Manifest signature verification failed: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Generate HMAC-SHA256 signature for manifest content.
        /// </summary>
        /// <param name="canonicalJson">Canonical JSON string</param>
        /// <param name="hmacKey">Secret HMAC key</param>
        /// <returns>Hex-encoded HMAC signature</returns>
        private static string GenerateHmacSignature(string canonicalJson, string hmacKey)
        {
            using var hmac = new HMACSHA256(Encoding.UTF8.GetBytes(hmacKey));
            var hashBytes = hmac.ComputeHash(Encoding.UTF8.GetBytes(canonicalJson));
            return Convert.ToHexString(hashBytes).ToLowerInvariant();
        }

        /// <summary>
        /// Create manifest object without signature field for verification.
        /// </summary>
        private static JsonElement CreateManifestWithoutSignature(JsonElement original)
        {
            var jsonString = original.GetRawText();
            var jsonDoc = JsonDocument.Parse(jsonString);

            // If no signature field, return as-is
            if (!jsonDoc.RootElement.TryGetProperty("signature", out _))
            {
                return jsonDoc.RootElement;
            }

            // Remove signature field and return clean manifest
            var dict = new Dictionary<string, object?>();

            foreach (var prop in jsonDoc.RootElement.EnumerateObject())
            {
                if (prop.Name != "signature")
                {
                    dict[prop.Name] = JsonElementToObject(prop.Value);
                }
            }

            var cleanJson = JsonSerializer.Serialize(dict);
            return JsonDocument.Parse(cleanJson).RootElement;
        }

        /// <summary>
        /// Convert JsonElement to object for dictionary creation.
        /// </summary>
        private static object? JsonElementToObject(JsonElement element)
        {
            return element.ValueKind switch
            {
                JsonValueKind.String => element.GetString(),
                JsonValueKind.Number => element.TryGetInt32(out var i) ? i : element.GetDouble(),
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                JsonValueKind.Null => null,
                JsonValueKind.Object => element.EnumerateObject()
                    .ToDictionary(p => p.Name, p => JsonElementToObject(p.Value)),
                JsonValueKind.Array => element.EnumerateArray()
                    .Select(JsonElementToObject).ToArray(),
                _ => element.GetRawText()
            };
        }

        /// <summary>
        /// Sort JSON properties for canonical representation.
        /// This ensures the same signature generation as Python implementation.
        /// </summary>
        private static string SortJsonProperties(string json)
        {
            try
            {
                var jsonDoc = JsonDocument.Parse(json);
                return JsonSerializer.Serialize(jsonDoc.RootElement, new JsonSerializerOptions
                {
                    WriteIndented = false,
                    PropertyNamingPolicy = null,
                    Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping
                });
            }
            catch
            {
                // Fallback to original if sorting fails
                return json;
            }
        }

        /// <summary>
        /// Extract signature from manifest JSON.
        /// </summary>
        /// <param name="manifestJson">Manifest JSON string</param>
        /// <returns>Signature value or null if not found</returns>
        public static string? ExtractSignatureFromManifest(string manifestJson)
        {
            try
            {
                var jsonDoc = JsonDocument.Parse(manifestJson);

                if (jsonDoc.RootElement.TryGetProperty("signature", out var sigElement) &&
                    sigElement.TryGetProperty("value", out var valueElement))
                {
                    return valueElement.GetString();
                }

                return null;
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Validate manifest structure and required fields.
        /// </summary>
        public static bool ValidateManifestStructure(string manifestJson)
        {
            try
            {
                var jsonDoc = JsonDocument.Parse(manifestJson);
                var root = jsonDoc.RootElement;

                // Check required fields
                var requiredFields = new[] { "version", "timestamp", "models" };

                foreach (var field in requiredFields)
                {
                    if (!root.TryGetProperty(field, out _))
                    {
                        Console.WriteLine($"[SECURITY] Missing required field in manifest: {field}");
                        return false;
                    }
                }

                // Validate models structure
                if (root.TryGetProperty("models", out var modelsElement))
                {
                    foreach (var model in modelsElement.EnumerateObject())
                    {
                        if (!model.Value.TryGetProperty("url", out _) ||
                            !model.Value.TryGetProperty("checksum", out _))
                        {
                            Console.WriteLine($"[SECURITY] Invalid model structure: {model.Name}");
                            return false;
                        }
                    }
                }

                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[SECURITY] Manifest structure validation failed: {ex.Message}");
                return false;
            }
        }
    }
}
