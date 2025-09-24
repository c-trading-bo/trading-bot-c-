using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// Integrity and signing service for logs, models, and manifests
    /// Provides SHA-256 hashing and digital signing for production data integrity
    /// </summary>
    public class IntegritySigningService
    {
        private readonly ILogger<IntegritySigningService> _logger;
        private readonly RSA _signingKey;
        private readonly string _publicKeyPem;

        public IntegritySigningService(ILogger<IntegritySigningService> logger)
        {
            _logger = logger;
            
            // Initialize RSA key pair for signing (in production, load from secure key store)
            _signingKey = RSA.Create(2048);
            _publicKeyPem = Convert.ToBase64String(_signingKey.ExportRSAPublicKey());
            
            _logger.LogInformation("🔐 [INTEGRITY] Integrity signing service initialized with RSA-2048");
        }

        /// <summary>
        /// Calculate SHA-256 hash of a file
        /// </summary>
        public async Task<string> CalculateFileHashAsync(string filePath)
        {
            try
            {
                using var sha256 = SHA256.Create();
                using var fileStream = File.OpenRead(filePath);
                var hashBytes = await Task.Run(() => sha256.ComputeHash(fileStream));
                var hash = Convert.ToHexString(hashBytes).ToLowerInvariant();
                
                _logger.LogDebug("📋 [INTEGRITY] File hash calculated: {File} -> {Hash}", 
                    Path.GetFileName(filePath), hash[..12] + "...");
                    
                return hash;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "🚨 [INTEGRITY] Error calculating file hash: {File}", filePath);
                throw;
            }
        }

        /// <summary>
        /// Calculate SHA-256 hash of string content
        /// </summary>
        public string CalculateContentHash(string content)
        {
            try
            {
                using var sha256 = SHA256.Create();
                var contentBytes = Encoding.UTF8.GetBytes(content);
                var hashBytes = sha256.ComputeHash(contentBytes);
                return Convert.ToHexString(hashBytes).ToLowerInvariant();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "🚨 [INTEGRITY] Error calculating content hash");
                throw;
            }
        }

        /// <summary>
        /// Create a signed manifest for a set of files
        /// </summary>
        public async Task<SignedManifest> CreateSignedManifestAsync(string manifestName, IEnumerable<string> filePaths)
        {
            try
            {
                var manifest = new SignedManifest
                {
                    Name = manifestName,
                    CreatedAt = DateTime.UtcNow,
                    Version = "1.0",
                    Files = new Dictionary<string, FileIntegrity>()
                };

                // Calculate hash for each file
                foreach (var filePath in filePaths)
                {
                    if (File.Exists(filePath))
                    {
                        var fileInfo = new FileInfo(filePath);
                        var hash = await CalculateFileHashAsync(filePath);
                        
                        manifest.Files[Path.GetFileName(filePath)] = new FileIntegrity
                        {
                            Path = filePath,
                            Hash = hash,
                            Size = fileInfo.Length,
                            LastModified = fileInfo.LastWriteTimeUtc,
                            HashAlgorithm = "SHA256"
                        };
                    }
                    else
                    {
                        _logger.LogWarning("⚠️ [INTEGRITY] File not found for manifest: {File}", filePath);
                    }
                }

                // Create manifest content
                var manifestJson = JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true });
                manifest.ContentHash = CalculateContentHash(manifestJson);
                
                // Sign the manifest
                manifest.Signature = SignContent(manifestJson);
                manifest.PublicKey = _publicKeyPem;

                _logger.LogInformation("✅ [INTEGRITY] Created signed manifest '{Name}' with {Count} files", 
                    manifestName, manifest.Files.Count);

                return manifest;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "🚨 [INTEGRITY] Error creating signed manifest: {Name}", manifestName);
                throw;
            }
        }

        /// <summary>
        /// Verify a signed manifest
        /// </summary>
        public async Task<ManifestVerificationResult> VerifySignedManifestAsync(SignedManifest manifest)
        {
            var result = new ManifestVerificationResult
            {
                IsValid = false,
                VerifiedAt = DateTime.UtcNow,
                Errors = new List<string>()
            };

            try
            {
                // Reconstruct manifest content for signature verification
                var tempManifest = new SignedManifest
                {
                    Name = manifest.Name,
                    CreatedAt = manifest.CreatedAt,
                    Version = manifest.Version,
                    Files = manifest.Files,
                    ContentHash = manifest.ContentHash
                };

                var manifestJson = JsonSerializer.Serialize(tempManifest, new JsonSerializerOptions { WriteIndented = true });

                // Verify signature
                if (!VerifySignature(manifestJson, manifest.Signature, manifest.PublicKey))
                {
                    result.Errors.Add("Invalid manifest signature");
                    _logger.LogError("🚨 [INTEGRITY] Invalid signature for manifest: {Name}", manifest.Name);
                    return result;
                }

                // Verify content hash
                var expectedHash = CalculateContentHash(manifestJson);
                if (manifest.ContentHash != expectedHash)
                {
                    result.Errors.Add("Manifest content hash mismatch");
                    _logger.LogError("🚨 [INTEGRITY] Content hash mismatch for manifest: {Name}", manifest.Name);
                    return result;
                }

                // Verify individual file hashes
                var verifiedFiles = 0;
                var missingFiles = 0;
                var corruptFiles = 0;

                foreach (var fileEntry in manifest.Files)
                {
                    var filePath = fileEntry.Value.Path;
                    
                    if (!File.Exists(filePath))
                    {
                        missingFiles++;
                        result.Errors.Add($"Missing file: {fileEntry.Key}");
                        continue;
                    }

                    var currentHash = await CalculateFileHashAsync(filePath);
                    if (currentHash != fileEntry.Value.Hash)
                    {
                        corruptFiles++;
                        result.Errors.Add($"File integrity violation: {fileEntry.Key}");
                        _logger.LogError("🚨 [INTEGRITY] File hash mismatch: {File}", fileEntry.Key);
                        continue;
                    }

                    verifiedFiles++;
                }

                result.VerifiedFiles = verifiedFiles;
                result.MissingFiles = missingFiles;
                result.CorruptFiles = corruptFiles;
                result.IsValid = result.Errors.Count == 0;

                if (result.IsValid)
                {
                    _logger.LogInformation("✅ [INTEGRITY] Manifest verification successful: {Name} ({Count} files)", 
                        manifest.Name, verifiedFiles);
                }
                else
                {
                    _logger.LogError("🚨 [INTEGRITY] Manifest verification failed: {Name} ({Errors} errors)", 
                        manifest.Name, result.Errors.Count);
                }

                return result;
            }
            catch (Exception ex)
            {
                result.Errors.Add($"Verification error: {ex.Message}");
                _logger.LogError(ex, "🚨 [INTEGRITY] Error verifying manifest: {Name}", manifest.Name);
                return result;
            }
        }

        /// <summary>
        /// Sign trading logs with integrity guarantee
        /// </summary>
        public async Task<SignedLogEntry> SignLogEntryAsync(string logContent, string logSource)
        {
            try
            {
                var entry = new SignedLogEntry
                {
                    Content = logContent,
                    Source = logSource,
                    Timestamp = DateTime.UtcNow,
                    ContentHash = CalculateContentHash(logContent)
                };

                var entryJson = JsonSerializer.Serialize(entry, new JsonSerializerOptions { WriteIndented = true });
                entry.Signature = SignContent(entryJson);
                entry.PublicKey = _publicKeyPem;

                _logger.LogDebug("📝 [INTEGRITY] Signed log entry from {Source}", logSource);
                return entry;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "🚨 [INTEGRITY] Error signing log entry from {Source}", logSource);
                throw;
            }
        }

        /// <summary>
        /// Verify a signed log entry
        /// </summary>
        public bool VerifyLogEntry(SignedLogEntry logEntry)
        {
            try
            {
                var tempEntry = new SignedLogEntry
                {
                    Content = logEntry.Content,
                    Source = logEntry.Source,
                    Timestamp = logEntry.Timestamp,
                    ContentHash = logEntry.ContentHash
                };

                var entryJson = JsonSerializer.Serialize(tempEntry, new JsonSerializerOptions { WriteIndented = true });
                
                // Verify content hash
                var expectedHash = CalculateContentHash(logEntry.Content);
                if (logEntry.ContentHash != expectedHash)
                {
                    _logger.LogError("🚨 [INTEGRITY] Log entry content hash mismatch");
                    return false;
                }

                // Verify signature
                return VerifySignature(entryJson, logEntry.Signature, logEntry.PublicKey);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "🚨 [INTEGRITY] Error verifying log entry");
                return false;
            }
        }

        /// <summary>
        /// Create integrity hash for model files with metadata
        /// </summary>
        public async Task<ModelIntegrity> CreateModelIntegrityAsync(string modelPath, Dictionary<string, object> metadata)
        {
            try
            {
                var modelHash = await CalculateFileHashAsync(modelPath);
                var metadataJson = JsonSerializer.Serialize(metadata);
                var metadataHash = CalculateContentHash(metadataJson);

                var integrity = new ModelIntegrity
                {
                    ModelPath = modelPath,
                    ModelHash = modelHash,
                    MetadataHash = metadataHash,
                    Metadata = metadata,
                    CreatedAt = DateTime.UtcNow,
                    HashAlgorithm = "SHA256"
                };

                var integrityJson = JsonSerializer.Serialize(integrity, new JsonSerializerOptions { WriteIndented = true });
                integrity.Signature = SignContent(integrityJson);
                integrity.PublicKey = _publicKeyPem;

                _logger.LogInformation("🧠 [INTEGRITY] Created model integrity for {Model}", Path.GetFileName(modelPath));
                return integrity;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "🚨 [INTEGRITY] Error creating model integrity: {Model}", modelPath);
                throw;
            }
        }

        /// <summary>
        /// Sign content using RSA private key
        /// </summary>
        private string SignContent(string content)
        {
            var contentBytes = Encoding.UTF8.GetBytes(content);
            var signatureBytes = _signingKey.SignData(contentBytes, HashAlgorithmName.SHA256, RSASignaturePadding.Pkcs1);
            return Convert.ToBase64String(signatureBytes);
        }

        /// <summary>
        /// Verify signature using RSA public key
        /// </summary>
        private bool VerifySignature(string content, string signature, string publicKeyPem)
        {
            try
            {
                using var rsa = RSA.Create();
                var publicKeyBytes = Convert.FromBase64String(publicKeyPem);
                rsa.ImportRSAPublicKey(publicKeyBytes, out _);

                var contentBytes = Encoding.UTF8.GetBytes(content);
                var signatureBytes = Convert.FromBase64String(signature);

                return rsa.VerifyData(contentBytes, signatureBytes, HashAlgorithmName.SHA256, RSASignaturePadding.Pkcs1);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error verifying signature");
                return false;
            }
        }

        public void Dispose()
        {
            _signingKey?.Dispose();
        }
    }

    /// <summary>
    /// Signed manifest containing file integrity information
    /// </summary>
    public class SignedManifest
    {
        public string Name { get; set; } = string.Empty;
        public DateTime CreatedAt { get; set; }
        public string Version { get; set; } = string.Empty;
        public Dictionary<string, FileIntegrity> Files { get; set; } = new();
        public string ContentHash { get; set; } = string.Empty;
        public string Signature { get; set; } = string.Empty;
        public string PublicKey { get; set; } = string.Empty;
    }

    /// <summary>
    /// File integrity information
    /// </summary>
    public class FileIntegrity
    {
        public string Path { get; set; } = string.Empty;
        public string Hash { get; set; } = string.Empty;
        public long Size { get; set; }
        public DateTime LastModified { get; set; }
        public string HashAlgorithm { get; set; } = "SHA256";
    }

    /// <summary>
    /// Signed log entry for audit trail
    /// </summary>
    public class SignedLogEntry
    {
        public string Content { get; set; } = string.Empty;
        public string Source { get; set; } = string.Empty;
        public DateTime Timestamp { get; set; }
        public string ContentHash { get; set; } = string.Empty;
        public string Signature { get; set; } = string.Empty;
        public string PublicKey { get; set; } = string.Empty;
    }

    /// <summary>
    /// Model integrity information
    /// </summary>
    public class ModelIntegrity
    {
        public string ModelPath { get; set; } = string.Empty;
        public string ModelHash { get; set; } = string.Empty;
        public string MetadataHash { get; set; } = string.Empty;
        public Dictionary<string, object> Metadata { get; set; } = new();
        public DateTime CreatedAt { get; set; }
        public string HashAlgorithm { get; set; } = "SHA256";
        public string Signature { get; set; } = string.Empty;
        public string PublicKey { get; set; } = string.Empty;
    }

    /// <summary>
    /// Result of manifest verification
    /// </summary>
    public class ManifestVerificationResult
    {
        public bool IsValid { get; set; }
        public DateTime VerifiedAt { get; set; }
        public int VerifiedFiles { get; set; }
        public int MissingFiles { get; set; }
        public int CorruptFiles { get; set; }
        public List<string> Errors { get; set; } = new();
    }
}