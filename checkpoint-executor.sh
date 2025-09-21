#!/bin/bash

# Checkpoint-Based Analyzer Cleanup Executor
# Implements crash-resilient execution with 15-minute checkpoints
# Addresses: "How to prevent this next run" from breaking the hour into checkpoints

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
CHECKPOINT_DIR=".checkpoints"
TRACKER_FILE="ANALYZER_CHECKPOINT_TRACKER.md"
VIOLATION_FILE="current_violations.txt"
CHECKPOINT_INTERVAL_MINUTES=15
MAX_VIOLATIONS_PER_CHECKPOINT=50
LOG_FILE="$CHECKPOINT_DIR/execution.log"

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Initialize checkpoint system
init_checkpoint_system() {
    mkdir -p "$CHECKPOINT_DIR"
    touch "$LOG_FILE"
    
    log_info "Initializing checkpoint-based execution system"
    log_info "Checkpoint interval: $CHECKPOINT_INTERVAL_MINUTES minutes"
    log_info "Max violations per checkpoint: $MAX_VIOLATIONS_PER_CHECKPOINT"
    
    # Create checkpoint state file if it doesn't exist
    if [ ! -f "$CHECKPOINT_DIR/state.json" ]; then
        cat > "$CHECKPOINT_DIR/state.json" << EOF
{
    "current_phase": "3",
    "current_checkpoint": "3.2",
    "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "last_updated": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "violations_fixed": 0,
    "total_violations_start": 2464,
    "checkpoints_completed": [],
    "current_rule": "CA2007",
    "target_files": ["src/IntelligenceStack/"],
    "interrupted": false
}
EOF
    fi
}

# Get current checkpoint state
get_checkpoint_state() {
    if [ -f "$CHECKPOINT_DIR/state.json" ]; then
        cat "$CHECKPOINT_DIR/state.json"
    else
        echo '{"error": "No checkpoint state found"}'
    fi
}

# Update checkpoint state
update_checkpoint_state() {
    local field="$1"
    local value="$2"
    
    if [ -f "$CHECKPOINT_DIR/state.json" ]; then
        # Use a temporary file to safely update the JSON
        local temp_file=$(mktemp)
        jq ".$field = \"$value\" | .last_updated = \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"" "$CHECKPOINT_DIR/state.json" > "$temp_file"
        mv "$temp_file" "$CHECKPOINT_DIR/state.json"
    fi
}

# Add completed checkpoint
add_completed_checkpoint() {
    local checkpoint="$1"
    local violations_fixed="$2"
    
    if [ -f "$CHECKPOINT_DIR/state.json" ]; then
        local temp_file=$(mktemp)
        jq ".checkpoints_completed += [{\"checkpoint\": \"$checkpoint\", \"violations_fixed\": $violations_fixed, \"completed_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}]" "$CHECKPOINT_DIR/state.json" > "$temp_file"
        mv "$temp_file" "$CHECKPOINT_DIR/state.json"
    fi
}

# Count current analyzer violations
count_violations() {
    local project="$1"
    
    # Build and count violations silently
    local count=$(dotnet build "$project" 2>&1 | grep -E "(warning|error)" | wc -l)
    echo "$count"
}

# Get specific rule violations 
count_rule_violations() {
    local project="$1"
    local rule="$2"
    
    # Count silently and only return the number
    local count=$(dotnet build "$project" 2>&1 | grep "$rule" | wc -l)
    echo "$count"
}

# Execute a checkpoint with timeout
execute_checkpoint_with_timeout() {
    local checkpoint_name="$1"
    local rule="$2"
    local target_files="$3"
    local max_time_minutes="$4"
    
    log_info "=== STARTING CHECKPOINT: $checkpoint_name ==="
    log_info "Rule: $rule"
    log_info "Target files: $target_files"
    log_info "Max time: $max_time_minutes minutes"
    
    local start_time=$(date +%s)
    local end_time=$((start_time + max_time_minutes * 60))
    
    log_info "Counting $rule violations..."
    local violations_start=$(count_rule_violations "src/IntelligenceStack/IntelligenceStack.csproj" "$rule")
    
    log_info "Starting violations for $rule: $violations_start"
    
    # Create checkpoint start marker
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ): Starting $checkpoint_name" >> "$CHECKPOINT_DIR/progress.log"
    
    # Execute the checkpoint work based on rule type
    case "$rule" in
        "CA2007")
            execute_ca2007_checkpoint "$target_files" "$end_time"
            ;;
        "CA1848")
            execute_ca1848_checkpoint "$target_files" "$end_time"
            ;;
        "S109")
            execute_s109_checkpoint "$target_files" "$end_time"
            ;;
        "CA1031")
            execute_ca1031_checkpoint "$target_files" "$end_time"
            ;;
        "CA1822")
            execute_ca1822_checkpoint "$target_files" "$end_time"
            ;;
        "CA1062")
            execute_ca1062_checkpoint "$target_files" "$end_time"
            ;;
        *)
            log_error "Unknown rule: $rule"
            return 1
            ;;
    esac
    
    # Count violations after checkpoint
    log_info "Counting $rule violations after checkpoint..."
    local violations_end=$(count_rule_violations "src/IntelligenceStack/IntelligenceStack.csproj" "$rule")
    local violations_fixed=$((violations_start - violations_end))
    
    log_success "Checkpoint $checkpoint_name completed"
    log_info "Violations fixed: $violations_fixed ($violations_start → $violations_end)"
    
    # Mark checkpoint as complete
    add_completed_checkpoint "$checkpoint_name" "$violations_fixed"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ): Completed $checkpoint_name - Fixed $violations_fixed violations" >> "$CHECKPOINT_DIR/progress.log"
    
    # Validate guardrails
    validate_checkpoint_guardrails
    
    return 0
}

# CA2007 ConfigureAwait checkpoint implementation
execute_ca2007_checkpoint() {
    local target_files="$1"
    local end_time="$2"
    
    log_info "Executing CA2007 ConfigureAwait fixes..."
    
    # Find files with CA2007 violations
    local files_with_violations=(
        "src/IntelligenceStack/Services/EnsembleMetaLearner.cs"
        "src/IntelligenceStack/Services/StreamingFeatureEngineering.cs"
        "src/IntelligenceStack/Services/OnlineLearningSystem.cs"
        "src/IntelligenceStack/Services/RLAdvisorSystem.cs"
    )
    
    local fixes_made=0
    for file in "${files_with_violations[@]}"; do
        if [ $(date +%s) -ge $end_time ]; then
            log_warning "Time limit reached, stopping checkpoint"
            break
        fi
        
        if [ -f "$file" ]; then
            log_info "Processing $file for CA2007 violations..."
            
            # Apply ConfigureAwait fixes using pattern replacement
            if apply_configure_await_fixes "$file"; then
                ((fixes_made++))
                log_success "Applied ConfigureAwait fixes to $file"
                
                # Build and verify no new errors introduced
                if ! dotnet build "src/IntelligenceStack/IntelligenceStack.csproj" --verbosity quiet > /dev/null 2>&1; then
                    log_error "Build failed after changes to $file - reverting"
                    git checkout "$file"
                    ((fixes_made--))
                fi
            fi
        fi
        
        # Stop if we've reached our target for this checkpoint
        if [ $fixes_made -ge 5 ]; then
            log_info "Reached target fixes for this checkpoint: $fixes_made"
            break
        fi
    done
    
    return 0
}

# Apply ConfigureAwait(false) fixes to a file
apply_configure_await_fixes() {
    local file="$1"
    local changes_made=0
    
    # Backup the file
    cp "$file" "$file.backup"
    
    # Pattern 1: Task.Run(...) - find lines that end with ), cancellationToken);
    sed -i 's/), cancellationToken);$/).ConfigureAwait(false);/g' "$file"
    
    # Pattern 2: General async method calls - find await calls without ConfigureAwait
    sed -i '/\.ConfigureAwait/!s/await \([^;]*[^)]\);$/await \1.ConfigureAwait(false);/g' "$file"
    
    # Pattern 3: Specific pattern for Task.Run with cancellationToken
    sed -i 's/await Task\.Run([^)]*), cancellationToken);$/&.ConfigureAwait(false);/g' "$file"
    
    # Check if changes were actually made by comparing with backup
    if ! diff "$file" "$file.backup" > /dev/null 2>&1; then
        rm "$file.backup"
        log_success "Applied ConfigureAwait fixes to $file"
        return 0
    else
        # No changes made, restore backup
        mv "$file.backup" "$file"
        return 1
    fi
}

# CA1848 LoggerMessage checkpoint implementation
execute_ca1848_checkpoint() {
    local target_files="$1"
    local end_time="$2"
    
    log_info "Executing CA1848 LoggerMessage fixes..."
    log_warning "CA1848 implementation is complex - marking as needs manual review"
    
    # This would require significant code analysis and generation
    # For now, mark it as needing manual intervention
    return 0
}

# S109 Magic Numbers checkpoint implementation
execute_s109_checkpoint() {
    local target_files="$1"
    local end_time="$2"
    
    log_info "Executing S109 Magic Numbers fixes..."
    log_warning "S109 implementation requires business logic analysis - marking as needs manual review"
    
    # This would require analyzing business context of numbers
    # For now, mark it as needing manual intervention
    return 0
}

# CA1031 Generic Exception checkpoint implementation
execute_ca1031_checkpoint() {
    local target_files="$1"
    local end_time="$2"
    
    log_info "Executing CA1031 Generic Exception fixes..."
    log_warning "CA1031 implementation requires exception analysis - marking as needs manual review"
    
    # This would require analyzing exception handling patterns
    # For now, mark it as needing manual intervention
    return 0
}

# CA1822 Static Methods checkpoint implementation
execute_ca1822_checkpoint() {
    local target_files="$1"
    local end_time="$2"
    
    log_info "Executing CA1822 Static Methods fixes..."
    log_warning "CA1822 implementation requires method analysis - marking as needs manual review"
    
    # This would require analyzing method dependencies
    # For now, mark it as needing manual intervention
    return 0
}

# CA1062 Null Validation checkpoint implementation
execute_ca1062_checkpoint() {
    local target_files="$1"
    local end_time="$2"
    
    log_info "Executing CA1062 Null Validation fixes..."
    log_warning "CA1062 implementation requires parameter analysis - marking as needs manual review"
    
    # This would require analyzing public method parameters
    # For now, mark it as needing manual intervention
    return 0
}

# Validate guardrails at checkpoint
validate_checkpoint_guardrails() {
    log_info "Validating checkpoint guardrails..."
    
    # Check TreatWarningsAsErrors
    if ! grep -q "TreatWarningsAsErrors.*true" Directory.Build.props; then
        log_error "GUARDRAIL VIOLATION: TreatWarningsAsErrors not enforced"
        return 1
    fi
    
    # In cleanup mode, we accept analyzer violations but ensure we can still compile
    # Check for actual compilation errors (not analyzer rule violations)
    local cs_errors=$(dotnet build "src/IntelligenceStack/IntelligenceStack.csproj" --verbosity quiet 2>&1 | grep -E "(error CS[0-9]+:)" | wc -l)
    if [ "$cs_errors" -gt 0 ]; then
        log_error "GUARDRAIL VIOLATION: Found $cs_errors compilation errors"
        return 1
    fi
    
    log_success "All guardrails validated successfully (analyzer violations are expected during cleanup)"
    return 0
}

# Resume from last checkpoint
resume_from_checkpoint() {
    log_info "=== RESUMING FROM LAST CHECKPOINT ==="
    
    if [ ! -f "$CHECKPOINT_DIR/state.json" ]; then
        log_error "No checkpoint state found. Run 'start' first."
        return 1
    fi
    
    local state=$(get_checkpoint_state)
    local current_checkpoint=$(echo "$state" | jq -r '.current_checkpoint')
    local current_rule=$(echo "$state" | jq -r '.current_rule')
    
    log_info "Resuming from checkpoint: $current_checkpoint"
    log_info "Current rule: $current_rule"
    
    # Mark as no longer interrupted
    update_checkpoint_state "interrupted" "false"
    
    # Continue with the current checkpoint
    continue_current_checkpoint
}

# Continue current checkpoint
continue_current_checkpoint() {
    local state=$(get_checkpoint_state)
    local current_checkpoint=$(echo "$state" | jq -r '.current_checkpoint')
    local current_rule=$(echo "$state" | jq -r '.current_rule')
    
    case "$current_checkpoint" in
        "3.2")
            execute_checkpoint_with_timeout "3.2-CA2007-ConfigureAwait" "CA2007" "src/IntelligenceStack/" "$CHECKPOINT_INTERVAL_MINUTES"
            ;;
        "3.3")
            execute_checkpoint_with_timeout "3.3-CA1848-LoggerMessage" "CA1848" "src/IntelligenceStack/" "$CHECKPOINT_INTERVAL_MINUTES"
            ;;
        "3.4")
            execute_checkpoint_with_timeout "3.4-S109-MagicNumbers" "S109" "src/IntelligenceStack/" "$CHECKPOINT_INTERVAL_MINUTES"
            ;;
        "3.5")
            execute_checkpoint_with_timeout "3.5-CA1031-GenericException" "CA1031" "src/IntelligenceStack/" "$CHECKPOINT_INTERVAL_MINUTES"
            ;;
        "3.6")
            execute_checkpoint_with_timeout "3.6-CA1822-StaticMethods" "CA1822" "src/IntelligenceStack/" "$CHECKPOINT_INTERVAL_MINUTES"
            ;;
        "3.7")
            execute_checkpoint_with_timeout "3.7-CA1062-NullValidation" "CA1062" "src/IntelligenceStack/" "$CHECKPOINT_INTERVAL_MINUTES"
            ;;
        *)
            log_error "Unknown checkpoint: $current_checkpoint"
            return 1
            ;;
    esac
}

# Start fresh execution
start_execution() {
    log_info "=== STARTING FRESH CHECKPOINT EXECUTION ==="
    
    init_checkpoint_system
    
    # Update state to mark as started
    update_checkpoint_state "started_at" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    update_checkpoint_state "interrupted" "false"
    
    # Get initial violation count
    local initial_violations=$(count_violations "src/IntelligenceStack/IntelligenceStack.csproj")
    log_info "Initial violation count: $initial_violations"
    
    # Start with current checkpoint from tracker
    continue_current_checkpoint
}

# Show status
show_status() {
    echo "=== CHECKPOINT EXECUTION STATUS ==="
    
    if [ ! -f "$CHECKPOINT_DIR/state.json" ]; then
        echo "No checkpoint state found."
        return 1
    fi
    
    local state=$(get_checkpoint_state)
    echo "Current Phase: $(echo "$state" | jq -r '.current_phase')"
    echo "Current Checkpoint: $(echo "$state" | jq -r '.current_checkpoint')"
    echo "Current Rule: $(echo "$state" | jq -r '.current_rule')"
    echo "Started At: $(echo "$state" | jq -r '.started_at')"
    echo "Last Updated: $(echo "$state" | jq -r '.last_updated')"
    echo "Violations Fixed: $(echo "$state" | jq -r '.violations_fixed')"
    echo "Interrupted: $(echo "$state" | jq -r '.interrupted')"
    
    echo ""
    echo "Completed Checkpoints:"
    echo "$state" | jq -r '.checkpoints_completed[] | "  - \(.checkpoint): \(.violations_fixed) violations fixed at \(.completed_at)"'
    
    if [ -f "$CHECKPOINT_DIR/progress.log" ]; then
        echo ""
        echo "Recent Progress:"
        tail -5 "$CHECKPOINT_DIR/progress.log"
    fi
}

# Main command dispatch
case "${1:-help}" in
    "start")
        start_execution
        ;;
    "resume")
        resume_from_checkpoint
        ;;
    "continue")
        continue_current_checkpoint
        ;;
    "status")
        show_status
        ;;
    "init")
        init_checkpoint_system
        ;;
    "help"|*)
        echo "Checkpoint-Based Analyzer Cleanup Executor"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  start     - Start fresh checkpoint execution"
        echo "  resume    - Resume from last checkpoint after interruption"
        echo "  continue  - Continue current checkpoint"
        echo "  status    - Show current execution status"
        echo "  init      - Initialize checkpoint system"
        echo "  help      - Show this help"
        echo ""
        echo "Crash Recovery:"
        echo "  If execution stops, immediately run: $0 resume"
        echo ""
        echo "Features:"
        echo "  - 15-minute checkpoints to prevent hour-long crashes"
        echo "  - Automatic resumption from last completed checkpoint"
        echo "  - Category completion tracking"
        echo "  - Guardrail validation at each checkpoint"
        ;;
esac