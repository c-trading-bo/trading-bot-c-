# Full-Stack Quality Gate

This directory contains the comprehensive quality gate system for the trading bot repository.

## Overview

The quality gate enforces these standards across the entire codebase:

### 🏗️ **Static Analysis (Currently Active)**

1. **Analyzer Compliance** - Zero tolerance for analyzer violations
   - Roslyn analyzers with `/warnaserror` enabled
   - SonarAnalyzer rules for code quality
   - Security analyzers for vulnerability detection
   - AsyncFixer for async/await pattern optimization

2. **Guardrail Enforcement** - Production readiness checks
   - **TODO/STUB/PLACEHOLDER Scan** - No placeholders in production code
   - **Commented Code Detection** - No commented-out production logic
   - **Hardcoded Credentials** - No secrets in source code
   - **Hardcoded URLs** - Environment-driven configuration required

3. **Security Pattern Scanning** - Anti-pattern detection
   - SSL/TLS validation bypasses
   - SQL injection patterns
   - Development URLs in production code

### 🔄 **Dead Code Detection (Framework Ready)**

- **CodeQL Integration Ready** - `.github/codeql/dead-code.ql` query prepared
- **Entry Point Analysis** - Detects unreachable code from orchestrator entry points
- **Call Graph Analysis** - Identifies unused methods, classes, and fields

## Workflow: `quality-gate.yml`

### Trigger Events
- **All Pushes** - Every commit is validated
- **All Pull Requests** - PR validation before merge

### Quality Checks

#### ✅ **Analyzer Build**
```bash
dotnet build --configuration Release --no-restore /warnaserror
```

#### ✅ **Placeholder Scan**
```bash
grep -RInE "TODO|STUB|PLACEHOLDER|NotImplementedException" \
  --exclude-dir={.git,.github,bin,obj} \
  --exclude="*.md" . | grep -v "TODO items"
```

#### ✅ **Commented Code Detection**
```bash
find . -name "*.cs" -exec grep -HnE "^[[:space:]]*//[[:space:]]*[A-Za-z_][A-Za-z0-9_]*[[:space:]]*[(].*[)].*;" {} \;
```

#### ✅ **Security Scanning**
- Hardcoded credentials pattern detection
- SSL/TLS bypass detection
- SQL injection pattern analysis

#### 🔧 **Dead Code Framework** (Ready for Integration)
```bash
# To enable CodeQL dead code detection:
# 1. Enable CodeQL in repository settings
# 2. Add CodeQL workflow with custom query
# 3. Reference .github/codeql/dead-code.ql
```

## Integration Guide

### Adding the Quality Gate to Your Branch

1. **Automatic Activation** - Runs on all pushes and PRs
2. **Local Testing** - Run checks locally before pushing:
   ```bash
   # Test analyzer compliance
   dotnet build --configuration Release /warnaserror
   
   # Test placeholder scan
   grep -RInE "TODO|STUB|PLACEHOLDER" --exclude-dir=.git src/
   
   # Test commented code scan
   find . -name "*.cs" -exec grep -HnE "^[[:space:]]*//.*[(].*[)].*;" {} \;
   ```

### Extending the Quality Gate

#### Adding New Security Patterns
Edit `.github/workflows/quality-gate.yml` security scan section:
```yaml
- name: Security pattern scan
  run: |
    # Add new security pattern checks
    if find ./src -name "*.cs" -exec grep -l "DANGEROUS_PATTERN" {} \;; then
      echo "❌ Found dangerous pattern"
      exit 1
    fi
```

#### Integrating Dead Code Detection
1. **Enable CodeQL** in repository settings
2. **Add CodeQL Workflow**:
   ```yaml
   - name: Initialize CodeQL
     uses: github/codeql-action/init@v2
     with:
       languages: csharp
       queries: .github/codeql/dead-code.ql
   ```

#### Custom Analyzer Rules
Add custom analyzers to `Directory.Build.props`:
```xml
<PackageReference Include="YourCustomAnalyzer" Version="1.0.0" PrivateAssets="all" />
```

## Maintenance

### Updating Exclusion Patterns

**Legitimate TODO References** - Add to exclusion list:
```bash
grep -v "TODO items" | grep -v "test results" | grep -v "NO TODO"
```

**Legitimate Comments** - Adjust commented code regex:
```bash
grep -v "Copyright" | grep -v "License" | grep -v "Description:"
```

### Performance Tuning

- **Parallel Execution** - Scans run in parallel where possible
- **Smart Exclusions** - Skip binary directories and generated files
- **Targeted Patterns** - Precise regex patterns to minimize false positives

## Status Dashboard

Track quality gate metrics:
- **Analyzer Violations**: Target 0 across all projects
- **Security Issues**: Target 0 critical/high severity
- **Dead Code %**: Target <5% of total codebase
- **Technical Debt**: Track TODO/placeholder trends

## Troubleshooting

### Common Issues

1. **False Positive Commented Code**
   - **Issue**: Legitimate comments flagged as code
   - **Fix**: Add to exclusion pattern in workflow

2. **Analyzer Rule Conflicts**
   - **Issue**: New analyzer rules conflict with existing code
   - **Fix**: Individual suppressions with justification in code

3. **Performance Impact**
   - **Issue**: Quality gate taking too long
   - **Fix**: Optimize scan patterns and add parallel execution

### Emergency Bypass

**Critical Hotfixes Only** - Add `[skip quality-gate]` to commit message:
```bash
git commit -m "Critical hotfix: Fix production outage [skip quality-gate]"
```

## Future Enhancements

- **📊 Quality Metrics Dashboard** - GitHub Pages integration
- **🔄 Auto-remediation** - Automated fixes for common issues  
- **📈 Trend Analysis** - Quality improvement tracking over time
- **🎯 Custom Rules** - Trading-specific analyzer rules