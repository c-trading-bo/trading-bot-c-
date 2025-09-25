#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE_REF:-origin/main}"
HEAD="${HEAD_REF:-HEAD}"
mkdir -p tools/analyzers

echo "=== Posture scan (.editorconfig / props / csproj) ==="
git diff "$BASE" "$HEAD" -- .editorconfig '**/*.editorconfig' '**/*.props' '**/*.targets' '**/*.csproj' | awk '/^\+/{print}' |
grep -Ei 'dotnet_(diagnostic|analyzer_diagnostic)\..*severity\s*=\s*(none|silent|hidden|suggestion|info|warning)|TreatWarningsAsErrors\s*=\s*false|EnableNETAnalyzers\s*=\s*false|RunAnalyzers.*=\s*false|<NoWarn>' \
&& { echo "❌ ALERT: Analyzer posture changed"; exit 2; } || echo "✅ Posture clean"

echo "=== Code suppressions (added lines only) ==="
git diff "$BASE" "$HEAD" -- '**/*.cs' | awk '/^\+/{print}' |
grep -En '#pragma\s+warning\s+disable|\[SuppressMessage\(|GlobalSuppressions\.cs' \
&& { echo "❌ ALERT: New suppressions"; exit 2; } || echo "✅ No suppressions"

echo "=== New public setters introduced (added lines only) ==="
git diff "$BASE" "$HEAD" -- '*.cs' | awk '/^\+/{print}' |
grep -En '\bpublic\s+[^\{]+\{\s*get;\s*set;\s*\}\s*$' || echo "OK: no new public setters"

echo "=== Collection setters (high risk) ==="
git diff "$BASE" "$HEAD" -- '*.cs' | awk '/^\+/{print}' |
grep -En '(List|Dictionary|I(ReadOnly)?(List|Dictionary))<.*>\s*\{\s*get;\s*set;' \
&& { echo "❌ ALERT: Collection properties made settable"; exit 2; } || echo "✅ No collection setters"

echo "=== Build (no fail-on-warning) → SARIF ==="
dotnet build -warnaserror- /p:errorlog=tools/analyzers/current.sarif >/dev/null || true

echo "=== Top analyzer/CS counts ==="
if command -v jq >/dev/null; then
  jq -r '.runs[0].results[]?.ruleId' tools/analyzers/current.sarif | sort | uniq -c | sort -nr | head -n 30
fi

echo "=== Targeted reliability/async/dispose snapshot ==="
if command -v jq >/dev/null; then
  jq -r '.runs[0].results[] | [.ruleId, (.locations[0].physicalLocation.artifactLocation.uri // ""), (.locations[0].physicalLocation.region.startLine // 0)] | @tsv' tools/analyzers/current.sarif |
  awk -F'\t' '/AsyncFixer0[1-5]|CA2000|CA1062|S2259|S2583|S2139|S112|S3904|S101|CA1848|CA2254|CA2227|CS0200|S109/'
fi