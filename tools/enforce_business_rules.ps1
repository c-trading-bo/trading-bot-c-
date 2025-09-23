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
        $_.FullName -notmatch '\\bin\\|\\obj\\|\\packages\\|\\test[^\\]*\\|\\Test[^\\]*\\|\\mock[^\\]*\\|\\Mock[^\\]*\\|\\simulation[^\\]*\\|\\Simulation[^\\]*\\' -and
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
    if ($files | Select-String -Pattern $Pattern -Quiet) {
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
        $excludeAI = '\\src\\IntelligenceStack\\|\\src\\OrchestratorAgent\\|\\src\\UnifiedOrchestrator\\|\\src\\BotCore\\|\\src\\RLAgent\\'
        Fail-IfMatch -Pattern '(Confidence|confidence)\s*[:=]\s*[0-9]+(\.[0-9]+)?[^0-9f]' -Message 'CRITICAL: ANY hardcoded AI confidence detected in production-critical code. Live trading forbidden.' -ExcludePattern $excludeAI

        # 5) ANY hardcoded position sizing outside critical code paths
        $excludePos = '\\src\\IntelligenceStack\\|\\src\\OrchestratorAgent\\|\\src\\UnifiedOrchestrator\\|\\src\\BotCore\\|\\src\\RLAgent\\|\\src\\ML\\|\\src\\Safety\\Tests\\'
        Fail-IfMatch -Pattern '(PositionSize|positionSize|Position|position)\s*[:=]\s*[0-9]+(\.[0-9]+)?[^0-9f]' -Message 'CRITICAL: ANY hardcoded position sizing detected. Live trading forbidden.' -ExcludePattern $excludePos

        # 6) ANY hardcoded thresholds/limits outside critical code paths
        $excludeThresh = '\\src\\IntelligenceStack\\|\\src\\OrchestratorAgent\\|\\src\\UnifiedOrchestrator\\|\\src\\BotCore\\|\\src\\RLAgent\\|\\src\\ML\\|\\src\\Safety\\Tests\\|\\src\\Strategies\\'
        Fail-IfMatch -Pattern '(Threshold|threshold|Limit|limit)\s*[:=]\s*[0-9]+(\.[0-9]+)?[^0-9f]' -Message 'CRITICAL: ANY hardcoded thresholds or limits detected. Live trading forbidden.' -ExcludePattern $excludeThresh
    }
    'Production' {
        # 1) Placeholder/mock words
        Fail-IfMatch -Pattern 'PLACEHOLDER|TEMP|DUMMY|MOCK|FAKE|STUB|HARDCODED|SAMPLE|placeholder|stub|fake|temporary|demo|example' -Message 'PRODUCTION VIOLATION: Mock/placeholder/stub patterns detected. All code must be production-ready.'

        # 2) Empty/placeholder async implementations
        Fail-IfMatch -Pattern 'Task\.Yield\(\)|Task\.Delay\([0-9]+\)|throw\s+new\s+NotImplementedException|return\s+Task\.CompletedTask\s*;' -Message 'PRODUCTION VIOLATION: Empty/placeholder async implementations detected.'

        # 3) Development/testing comments
        Fail-IfMatch -Pattern '///\s*(TODO|FIXME|HACK|XXX|STUB|PLACEHOLDER|BUG|NOTE|REVIEW|REFACTOR|OPTIMIZE|for\s+testing|debug\s+only|temporary|remove\s+this|implementation\s+needed|production\s+replacement|real\s+data\s+needed|configuration\s+required)' -Message 'PRODUCTION VIOLATION: Development/testing comments detected.'

        # 4) Weak RNG usage
        Fail-IfMatch -Pattern 'new\s+Random\(\)|Random\.Shared|_random\.Next|\.NextDouble\(\)' -Message 'CRITICAL: Weak random number generation detected. Security violation.'
    }
}

Write-Host "Guardrail '$Mode' checks passed."
