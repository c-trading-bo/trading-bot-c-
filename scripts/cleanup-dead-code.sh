#!/bin/bash

# Dead-code audit & cleanup script
# Moves unused files to archive/ directory for safe deletion via PR
# Identified targets: legacy dashboards, unused views, StandaloneDashboard

echo "🗑️  Starting dead-code audit & cleanup..."

# Create archive directory for unused files
mkdir -p archive/legacy-dashboards
mkdir -p archive/unused-components

# Move Dashboard module (no references found)
echo "📁 Moving unused Dashboard module..."
if [ -d "src/Dashboard" ]; then
    mv src/Dashboard archive/legacy-dashboards/
    echo "✅ Moved src/Dashboard to archive/legacy-dashboards/"
fi

# Move StandaloneDashboard (no references found)
echo "📁 Moving StandaloneDashboard..."
if [ -d "src/StandaloneDashboard" ]; then
    mv src/StandaloneDashboard archive/legacy-dashboards/
    echo "✅ Moved src/StandaloneDashboard to archive/legacy-dashboards/"
fi

# Move potentially unused wwwroot files
echo "📁 Archiving legacy dashboard HTML files..."
if [ -f "wwwroot/auto-background-dashboard.html" ]; then
    mv wwwroot/auto-background-dashboard.html archive/legacy-dashboards/
    echo "✅ Moved auto-background-dashboard.html to archive/"
fi

# Create README for archived files
cat > archive/README.md << 'EOF'
# Archived Files - Dead Code Audit

This directory contains files identified as unused during the dead-code audit.

## Legacy Dashboards
- `legacy-dashboards/Dashboard/` - Unused dashboard module with no references
- `legacy-dashboards/StandaloneDashboard/` - Unused standalone dashboard with no references  
- `legacy-dashboards/auto-background-dashboard.html` - Unused HTML dashboard

## Analysis Results
- No references found to these components in the codebase
- Safe to delete after review
- Moved here instead of direct deletion for safety

## Next Steps
1. Review archived files to ensure they're truly unused
2. Delete archive/ directory via PR if confirmed unused
3. Update any documentation that references these components

Date: $(date)
Action: Dead-code audit cleanup
EOF

echo "📊 Dead-code audit summary:"
echo "  - Archived Dashboard module (no references)"
echo "  - Archived StandaloneDashboard (no references)"
echo "  - Archived legacy HTML files"
echo "  - Created archive/README.md with analysis"

echo "✅ Dead-code audit completed. Review archive/ directory before deletion."