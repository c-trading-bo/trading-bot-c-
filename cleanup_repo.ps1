$ErrorActionPreference = "Stop"
$root = "C:\Users\kevin\Downloads\C# ai bot"
Set-Location $root

# === PREVIEW WHAT WILL BE DELETED ===
"=== Preview: directories to delete (artifacts) ==="
$artifactDirs = Get-ChildItem -Recurse -Force -Directory |
  Where-Object { $_.Name -in @('bin','obj','.vs','packages','TestResults') }
$artifactDirs | Select-Object -ExpandProperty FullName | Sort-Object

"`n=== Preview: files to delete (junk patterns) ==="
$junkFiles = Get-ChildItem -Recurse -Force -File -Include `
  *.user, *.suo, *.cache, *.tmp, *.log, '.DS_Store', 'Thumbs.db'
$junkFiles | Select-Object -ExpandProperty FullName | Sort-Object

# === DELETE ARTIFACTS ===
if ($artifactDirs) {
  $artifactDirs | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
  Write-Host "`nArtifacts deleted."
} else { Write-Host "`nNo artifact directories found."; }

if ($junkFiles) {
  $junkFiles | Remove-Item -Force -ErrorAction SilentlyContinue
  Write-Host "Junk files deleted."
} else { Write-Host "No junk files found."; }

# === OPTIONAL: delete legacy root files if you're moving to the new structure ===
$optional = @(
  ".\C# ai bot.csproj",
  ".\C# ai bot.sln",
  ".\Program.cs",
  ".\PlaceOrderAndWatchFill.cs"
)
"`n=== Optional removals (uncomment next block to execute) ==="
$optional | ForEach-Object { $_ }

# To actually delete the optional items, remove the <# #> comment block below:
<#
$optional | ForEach-Object {
  if (Test-Path $_) { Remove-Item $_ -Force -ErrorAction SilentlyContinue }
}
Write-Host "Optional legacy files removed."
#>
