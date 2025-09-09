#!/bin/bash

# Start Decision Service
echo "🚀 Starting ML/RL Decision Service..."

# Set environment variables
export DECISION_SERVICE_HOST=${DECISION_SERVICE_HOST:-127.0.0.1}
export DECISION_SERVICE_PORT=${DECISION_SERVICE_PORT:-7080}
export DECISION_SERVICE_CONFIG=${DECISION_SERVICE_CONFIG:-decision_service_config.yaml}

echo "📊 Service will start on ${DECISION_SERVICE_HOST}:${DECISION_SERVICE_PORT}"
echo "🔧 Using config: ${DECISION_SERVICE_CONFIG}"

# Change to decision service directory
cd "$(dirname "$0")"

# Check if config exists
if [ ! -f "$DECISION_SERVICE_CONFIG" ]; then
    echo "⚠️  Config file not found: $DECISION_SERVICE_CONFIG"
    echo "Using default configuration..."
fi

# Start the service
python decision_service.py