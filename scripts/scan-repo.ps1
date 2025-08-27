# Deep repo scan: duplicates, empty folders, and safe-to-clean items
# REPORT-ONLY. Does not delete files.
# Usage: powershell -ExecutionPolicy Bypass -File .\scripts\scan-repo.ps1

param(
    [string]$OutFile = "SCAN_REPORT.txt"
)

$ErrorActionPreference = 'Stop'
Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Definition) | Out-Null
Set-Location -Path (Resolve-Path '..')

function Hash-File($path) {
    try {
        return (Get-FileHash -Path $path -Algorithm SHA256).Hash
    } catch { return $null }
}

# Collect files, skipping common build artifacts to reduce noise in duplicate scan size, but still list them in categories
$allFiles = Get-ChildItem -Path . -Recurse -Force -File | Where-Object {
    $_.FullName -notmatch "\\.git\\" -and $_.FullName -notmatch "\\.vs\\"
}

# Categories for potential cleanup
$categories = @{
    Build = [System.Collections.Generic.List[string]]::new()
    TestBuild = [System.Collections.Generic.List[string]]::new()
    Logs = [System.Collections.Generic.List[string]]::new()
    State = [System.Collections.Generic.List[string]]::new()
    Journal = [System.Collections.Generic.List[string]]::new()
    Examples = [System.Collections.Generic.List[string]]::new()
    Docs = [System.Collections.Generic.List[string]]::new()
    Scripts = [System.Collections.Generic.List[string]]::new()
    Source = [System.Collections.Generic.List[string]]::new()
    Other = [System.Collections.Generic.List[string]]::new()
}

function Add-Cat($file) {
    $p = $file.FullName
    if ($p -match "\\(bin|obj)\\") { if ($p -match "\\tests\\") { $categories.TestBuild.Add($p) } else { $categories.Build.Add($p) } return }
    if ($p -match "\\tests\\.*\\(bin|obj)\\") { $categories.TestBuild.Add($p); return }
    if ($p -match "\\journal\\") { $categories.Journal.Add($p); return }
    if ($p -match "\\state\\") { $categories.State.Add($p); return }
    if ($p -match "\\logs?\\" -or $p -match "\.log$") { $categories.Logs.Add($p); return }
    if ($p -match "\\examples\\") { $categories.Examples.Add($p); return }
    if ($p -match "\\docs\\") { $categories.Docs.Add($p); return }
    if ($p -match "\\scripts\\" -or $p -match "\.ps1$" -or $p -match "\.cmd$") { $categories.Scripts.Add($p); return }
    if ($p -match "\\src\\") { $categories.Source.Add($p); return }
    $categories.Other.Add($p)
}

$dupCandidates = @()
foreach ($f in $allFiles) {
    Add-Cat $f
    $dupCandidates += $f
}

# Compute hashes for duplicates (include everything except very large build outputs > 50 MB)
$hashMap = @{}
foreach ($f in $dupCandidates) {
    try {
        if ($f.Length -gt 50MB) { continue }
        $h = Hash-File $f.FullName
        if (-not $h) { continue }
        if (-not $hashMap.ContainsKey($h)) { $hashMap[$h] = New-Object System.Collections.Generic.List[string] }
        $hashMap[$h].Add($f.FullName)
    } catch { }
}

$duplicateGroups = $hashMap.GetEnumerator() | Where-Object { $_.Value.Count -gt 1 } | ForEach-Object {
    [PSCustomObject]@{ Hash = $_.Key; Files = ($_.Value | Sort-Object) }
} | Sort-Object { $_.Files[0] }

# Empty directories (no files under them)
$allDirs = Get-ChildItem -Path . -Recurse -Force -Directory | Where-Object { $_.FullName -notmatch "\\.git\\|\\.vs\\" }
$emptyDirs = @()
foreach ($d in $allDirs) {
    try {
        $hasFile = (Get-ChildItem -Path $d.FullName -Recurse -Force -File -ErrorAction SilentlyContinue | Select-Object -First 1)
        if (-not $hasFile) { $emptyDirs += $d.FullName }
    } catch { }
}

# Size summary helper
function Sum-Size($paths) {
    $bytes = 0
    foreach ($p in $paths) { try { $bytes += (Get-Item -LiteralPath $p).Length } catch { } }
    return $bytes
}
function Fmt-Size($bytes) {
    switch ($bytes) {
        {$_ -ge 1GB} { return "{0:N2} GB" -f ($bytes/1GB) }
        {$_ -ge 1MB} { return "{0:N2} MB" -f ($bytes/1MB) }
        {$_ -ge 1KB} { return "{0:N2} KB" -f ($bytes/1KB) }
        default { return "$bytes B" }
    }
}

# Compose report text
$sb = New-Object System.Text.StringBuilder
$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
$null = $sb.AppendLine("Deep Repository Scan (REPORT ONLY)")
$null = $sb.AppendLine("Generated: $ts")
$null = $sb.AppendLine("")

# Summary
$null = $sb.AppendLine("Summary")
$null = $sb.AppendLine("- Total files: {0}" -f $allFiles.Count)
$null = $sb.AppendLine("- Duplicate groups: {0}" -f ($duplicateGroups | Measure-Object).Count)
$null = $sb.AppendLine("- Empty directories: {0}" -f $emptyDirs.Count)
$null = $sb.AppendLine("")

# Category sizes
function Add-CategorySection($name, $list) {
    $size = 0
    foreach ($p in $list) { try { $size += (Get-Item -LiteralPath $p).Length } catch { } }
    $null = $sb.AppendLine(([string]::Format("- {0}: {1} files, {2}", $name, $list.Count, (Fmt-Size $size))))
}
$null = $sb.AppendLine("Sizes by category:")
Add-CategorySection "Build" $categories.Build
Add-CategorySection "TestBuild" $categories.TestBuild
Add-CategorySection "Logs" $categories.Logs
Add-CategorySection "State" $categories.State
Add-CategorySection "Journal" $categories.Journal
Add-CategorySection "Examples" $categories.Examples
Add-CategorySection "Docs" $categories.Docs
Add-CategorySection "Scripts" $categories.Scripts
Add-CategorySection "Source" $categories.Source
Add-CategorySection "Other" $categories.Other
$null = $sb.AppendLine("")

# Duplicates section
$null = $sb.AppendLine("Duplicate file groups (by SHA256):")
foreach ($g in $duplicateGroups) {
    $null = $sb.AppendLine("Hash: {0}" -f $g.Hash)
    foreach ($fp in $g.Files) { $null = $sb.AppendLine("  - {0}" -f $fp) }
}
if ((($duplicateGroups | Measure-Object).Count) -eq 0) { $null = $sb.AppendLine("(none)") }
$null = $sb.AppendLine("")

# Empty directories
$null = $sb.AppendLine("Empty directories:")
foreach ($ed in ($emptyDirs | Sort-Object)) { $null = $sb.AppendLine("  - {0}" -f $ed) }
if ($emptyDirs.Count -eq 0) { $null = $sb.AppendLine("(none)") }
$null = $sb.AppendLine("")

# Proposed cleanup (report only)
$null = $sb.AppendLine("Proposed cleanup (safe defaults) -- NO ACTION TAKEN:")
$null = $sb.AppendLine("- Remove build outputs: **/bin, **/obj")
$null = $sb.AppendLine("- Remove test outputs: tests/**/bin, tests/**/obj")
$null = $sb.AppendLine("- Remove logs: ./logs/*.log (if present)")
$null = $sb.AppendLine("- Trim journals: ./journal/ (keep latest N) -- manual decision recommended")
$null = $sb.AppendLine("- Remove empty directories listed above")
$null = $sb.AppendLine("- Review duplicates above; keep one canonical copy per group and remove/ignore the rest")
$null = $sb.AppendLine("")
$null = $sb.AppendLine("To execute cleanup after review, I can generate a cleanup script (e.g., scripts/clean-repo.ps1) that applies the agreed actions.")

# Write report (overwrite)
Set-Content -Path $OutFile -Value $sb.ToString() -Encoding UTF8

Write-Host "[scan-repo] Report written to $OutFile"