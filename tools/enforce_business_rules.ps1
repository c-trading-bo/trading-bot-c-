param(
    [ValidateSet('Business','Production')]
    [string]$Mode = 'Business'
)

$ErrorActionPreference = 'Stop'

function Get-CodeFiles {
    param(
        [string]$ExcludePattern
    )
    Get-ChildItem -Path . -Recurse -Include *.cs | Where-Object {
        $_.FullName -notmatch '[\\/]archive[\\/]|[\\/]bin[\\/]|[\\/]obj[\\/]|[\\/]packages[\\/]|[\\/]test[^\\/]*[\\/]|[\\/]Test[^\\/]*[\\/]|[\\/]mock[^\\/]*[\\/]|[\\/]Mock[^\\/]*[\\/]|[\\/]simulation[^\\/]*[\\/]|[\\/]Simulation[^\\/]*[\\/]' -and
        ($ExcludePattern -eq '' -or $_.FullName -notmatch $ExcludePattern)
    }
}

function Fail-IfMatch {
    param(
        [string]$Pattern,
        [string]$Message,
        [string]$ExcludePattern = ''
    )
    $files = Get-CodeFiles -ExcludePattern $ExcludePattern
    $matchResults = $files | Select-String -Pattern $Pattern -Quiet
    if ($matchResults -contains $true) {
        Write-Host $Message
        exit 1
    }
}

switch ($Mode) {
    'Business' {
        # 1) Specific hardcoded position sizing 2.5
        Fail-IfMatch -Pattern '(PositionSize|positionSize|Position|position).*[:=]\s*(2\.5)[^0-9f]' -Message 'CRITICAL: Hardcoded position sizing value 2.5 detected. Use MLConfigurationService.GetPositionSizeMultiplier() instead.'

        # 2) Specific hardcoded AI confidence 0.7
        Fail-IfMatch -Pattern '(Confidence|confidence).*[:=]\s*(0\.7)[^0-9f]' -Message 'CRITICAL: Hardcoded AI confidence value 0.7 detected. Use MLConfigurationService.GetAIConfidenceThreshold() instead.'

        # 3) Specific hardcoded regime detection 1.0
        Fail-IfMatch -Pattern '(Regime|regime).*[:=]\s*(1\.0)[^0-9f]' -Message 'CRITICAL: Hardcoded regime detection value 1.0 detected. Use MLConfigurationService.GetRegimeDetectionThreshold() instead.'

        # 4) ANY hardcoded AI confidence outside critical code paths
        $excludeAI = '[\\/]src[\\/]IntelligenceStack[\\/]|[\\/]src[\\/]OrchestratorAgent[\\/]|[\\/]src[\\/]UnifiedOrchestrator[\\/]|[\\/]src[\\/]BotCore[\\/]|[\\/]src[\\/]RLAgent[\\/]|[\\/]src[\\/]Backtest[\\/]'
        Fail-IfMatch -Pattern '(Confidence|confidence)\s*[:=]\s*[0-9]+(\.[0-9]+)?[^0-9f]' -Message 'CRITICAL: ANY hardcoded AI confidence detected in production-critical code. Live trading forbidden.' -ExcludePattern $excludeAI

        # 5) ANY hardcoded position sizing outside critical code paths
        $excludePos = '[\\/]src[\\/]IntelligenceStack[\\/]|[\\/]src[\\/]OrchestratorAgent[\\/]|[\\/]src[\\/]UnifiedOrchestrator[\\/]|[\\/]src[\\/]BotCore[\\/]|[\\/]src[\\/]RLAgent[\\/]|[\\/]src[\\/]ML[\\/]|[\\/]src[\\/]Safety[\\/]Tests[\\/]|[\\/]src[\\/]Backtest[\\/]'
        Fail-IfMatch -Pattern '(PositionSize|positionSize|Position|position)\s*[:=]\s*[0-9]+(\.[0-9]+)?[^0-9f]' -Message 'CRITICAL: ANY hardcoded position sizing detected. Live trading forbidden.' -ExcludePattern $excludePos

        # 6) ANY hardcoded thresholds/limits outside critical code paths
        $excludeThresh = '[\\/]src[\\/]IntelligenceStack[\\/]|[\\/]src[\\/]OrchestratorAgent[\\/]|[\\/]src[\\/]UnifiedOrchestrator[\\/]|[\\/]src[\\/]BotCore[\\/]|[\\/]src[\\/]RLAgent[\\/]|[\\/]src[\\/]ML[\\/]|[\\/]src[\\/]Safety[\\/]Tests[\\/]|[\\/]src[\\/]Strategies[\\/]|[\\/]src[\\/]Backtest[\\/]|[\\/]src[\\/]Abstractions[\\/]|[\\/]src[\\/]Monitoring[\\/]|[\\/]src[\\/]Zones[\\/]'
        Fail-IfMatch -Pattern '(Threshold|threshold|Limit|limit)\s*[:=]\s*[0-9]+(\.[0-9]+)?[^0-9f]' -Message 'CRITICAL: ANY hardcoded thresholds or limits detected. Live trading forbidden.' -ExcludePattern $excludeThresh
    }
    'Production' {
        # 1) Only catch ACTUAL placeholder code patterns, not documentation about replacing mocks
        # Pattern: Only flag comments that indicate incomplete code (e.g., "// Placeholder:", "// MOCK:", "// STUB:")
        # Exclude: Documentation explaining production implementations, SafetyAnalyzers, Tests, ML training code, BotCore (uses temp files)
        $excludeLegitimate = '[\\/]src[\\/]Safety[\\/]Analyzers[\\/]|[\\/]src[\\/]IntelligenceStack[\\/]|[\\/]src[\\/]Tests[\\/]|[\\/]src[\\/]Backtest[\\/]|[\\/]src[\\/]BotCore[\\/]|[\\/]src[\\/]UnifiedOrchestrator[\\/]'
        Fail-IfMatch -Pattern '//\s*(PLACEHOLDER|MOCK|STUB|FAKE|DUMMY|TEMP)[\s:]' -Message 'PRODUCTION VIOLATION: Placeholder code comments detected. All code must be production-ready.' -ExcludePattern $excludeLegitimate

        # 2) Empty/placeholder async implementations (actual problematic code)
        Fail-IfMatch -Pattern 'throw\s+new\s+NotImplementedException' -Message 'PRODUCTION VIOLATION: NotImplementedException detected.'

        # 3) Development task comments (TODO/FIXME/HACK with colon or space)
        Fail-IfMatch -Pattern '//\s*(TODO|FIXME|HACK|XXX)[\s:]' -Message 'PRODUCTION VIOLATION: Development task comments (TODO/FIXME/HACK) detected.'

        # 4) Weak RNG usage (excluding legitimate uses in Backtest/ML/Bandits)
        $excludeRNG = '[\\/]src[\\/]Backtest[\\/]|[\\/]src[\\/]IntelligenceStack[\\/]|[\\/]src[\\/]BotCore[\\/]Bandits[\\/]|[\\/]src[\\/]RLAgent[\\/]|[\\/]src[\\/]UnifiedOrchestrator[\\/]Runtime[\\/]'
        Fail-IfMatch -Pattern 'new\s+Random\(\)(?!\(.*GetHashCode)' -Message 'CRITICAL: Weak random number generation detected. Use cryptographic RNG or seeded Random for determinism.' -ExcludePattern $excludeRNG
    }
}

Write-Host "Guardrail '$Mode' checks passed."
