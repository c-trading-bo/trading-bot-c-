using System;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;

namespace BotCore.Services
{
    /// <summary>
    /// CI egress protection handler - blocks cloud API calls when CI=1
    /// Prevents accidental live API usage in CI environments while preserving local functionality
    /// </summary>
    public sealed class CloudEgressGuardHandler : DelegatingHandler
    {
        private readonly bool _ci;
        
        public CloudEgressGuardHandler(IConfiguration cfg) 
        {
            _ci = cfg.GetValue("CI", false);
        }
        
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct)
        {
            if (_ci) 
            {
                throw new InvalidOperationException("Cloud egress blocked (CI=1).");
            }
            return base.SendAsync(request, ct);
        }
    }
}