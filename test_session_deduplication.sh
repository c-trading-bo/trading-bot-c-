#!/bin/bash
# Test script to demonstrate session deduplication functionality

echo "🧪 Testing GitHub Actions Session Deduplication"
echo "================================================="

# Test 1: First session should be allowed
echo "🔍 Test 1: First session registration"
python .github/scripts/session_deduplicator.py check push "test-workflow" "test123" "run456"
RESULT1=$?
echo "Exit code: $RESULT1 (0=allowed, 1=prevented)"

# Register the session
echo ""
echo "📝 Registering session..."
python .github/scripts/session_deduplicator.py register "test-workflow-push-test123" "push" "test-workflow" "run456" "test123"

# Test 2: Duplicate session should be prevented
echo ""
echo "🔍 Test 2: Duplicate session prevention"
python .github/scripts/session_deduplicator.py check push "test-workflow" "test123" "run789"
RESULT2=$?
echo "Exit code: $RESULT2 (0=allowed, 1=prevented)"

# Test 3: Push+PR deduplication
echo ""
echo "🔍 Test 3: Push+PR deduplication test"

# First simulate a PR event  
python .github/scripts/session_deduplicator.py register "test-workflow-pull_request-abc789" "pull_request" "test-workflow" "pr123" "abc789"
echo "PR session registered"

# Then try a push event for the same commit - should be prevented
python .github/scripts/session_deduplicator.py check push "test-workflow" "abc789" "push456"
RESULT3=$?
echo "Push after PR exit code: $RESULT3 (0=allowed, 1=prevented)"

# Test 4: Clean up and verify
echo ""
echo "🧹 Test 4: Session cleanup"
python .github/scripts/session_deduplicator.py cleanup "test-workflow-push-test123"
python .github/scripts/session_deduplicator.py cleanup "test-workflow-pull_request-abc789"

# Test audit functionality
echo ""
echo "📊 Test 5: Audit logging"
python .github/scripts/session_deduplicator.py audit "test-audit-session" "workflow_run" "test-audit" "audit123" "audit789" "true" "success"

echo ""
echo "✅ Session deduplication tests completed!"
echo "Expected results:"
echo "  - Test 1: Allowed (exit code 0)"
echo "  - Test 2: Prevented (exit code 1) - duplicate session"  
echo "  - Test 3: Prevented (exit code 1) - push after PR"
echo "  - All syntax should be valid"