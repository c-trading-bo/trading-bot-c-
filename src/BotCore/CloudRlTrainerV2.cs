// CloudRlTrainerV2 has been moved to src/Cloud/CloudRlTrainerV2.cs
// This is a production-ready implementation with full analyzer compliance
// See src/Cloud/README.md for usage instructions

using System;

namespace BotCore
{
    /// <summary>
    /// CloudRlTrainerV2 has been relocated to src/Cloud/ for better organization.
    /// The new implementation includes:
    /// - Full analyzer compliance and production guardrails
    /// - Multi-source model download (GitHub, cloud storage, custom APIs)
    /// - Hot-swap capability with ONNX session management
    /// - Rate limiting, retry logic, and integrity verification
    /// - Performance tracking and automatic model selection
    /// - Complete dependency injection setup
    /// 
    /// Please use the CloudTrainer.CloudRlTrainerV2 class from src/Cloud/
    /// and follow the setup instructions in src/Cloud/README.md
    /// </summary>
    [Obsolete("Use CloudTrainer.CloudRlTrainerV2 from src/Cloud/ instead. See src/Cloud/README.md for setup.")]
    public class CloudRlTrainerV2
    {
        public CloudRlTrainerV2()
        {
            throw new InvalidOperationException(
                "CloudRlTrainerV2 has been moved to src/Cloud/CloudRlTrainerV2.cs. " +
                "Please use CloudTrainer.CloudRlTrainerV2 and follow setup instructions in src/Cloud/README.md");
        }
    }
}
