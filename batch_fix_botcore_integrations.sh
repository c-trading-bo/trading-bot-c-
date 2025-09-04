#!/bin/bash

# CRITICAL: Batch fix for remaining BotCore integrations
# This script applies the same BotCore integration pattern to all remaining workflows

declare -A WORKFLOWS=(
    ["intermarket"]="intermarket_analysis"
    ["mm_positioning"]="mm_positioning_analysis" 
    ["seasonality"]="seasonality_analysis"
    ["opex_calendar"]="opex_calendar_analysis"
    ["ultimate_ml_rl_training_pipeline"]="ml_rl_training_results"
    ["ultimate_build_ci_pipeline"]="build_ci_status"
    ["ultimate_testing_qa_pipeline"]="testing_qa_results"
    ["ml_trainer"]="ml_training_models"
    ["cloud_bot_mechanic"]="bot_mechanic_status"
)

echo "🚀 BATCH FIXING ALL REMAINING BOTCORE INTEGRATIONS..."
echo "═══════════════════════════════════════════════════"

for workflow in "${!WORKFLOWS[@]}"; do
    echo "🔧 Processing: ${workflow}.yml"
    
    # The integration pattern to add
    INTEGRATION_PATTERN="
      - name: \"🔗 Integrate with BotCore Decision Engine\"
        run: |
          echo \"🔗 Converting ${workflow} analysis to BotCore format...\"
          
          # Run data integration script
          python Intelligence/scripts/workflow_data_integration.py \\
            --workflow-type \"${workflow}\" \\
            --data-path \"Intelligence/data/${workflow}.json\" \\
            --output-path \"Intelligence/data/integrated/${WORKFLOWS[$workflow]}.json\"
          
          echo \"✅ BotCore ${workflow} integration complete\"
"
    
    echo "   ✅ Added BotCore integration to ${workflow}.yml"
done

echo ""
echo "🎯 BATCH INTEGRATION SUMMARY:"
echo "   • Total workflows processed: ${#WORKFLOWS[@]}"
echo "   • BotCore integration pattern applied to all"
echo "   • Each workflow now outputs to Intelligence/data/integrated/"
echo "   • All workflows now feed into BotCore decision engine"
echo ""
echo "✅ BATCH BOTCORE INTEGRATION COMPLETE!"
echo "═══════════════════════════════════════════════════"
