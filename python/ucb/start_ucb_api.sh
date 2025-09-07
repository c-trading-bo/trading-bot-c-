#!/bin/bash

# Production UCB API Startup Script
# Run this to start the Neural UCB FastAPI service

echo "🚀 Starting UCB FastAPI Service..."
echo "📁 Working directory: $(pwd)"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install/update requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Set environment variables for production
export UCB_HOST="0.0.0.0"
export UCB_PORT="5000"
export UCB_PERSISTENCE_PATH="ucb_state.pkl"
export UCB_WEIGHTS_PATH="neural_ucb_topstep.pth"

echo "🌐 Starting UCB service on ${UCB_HOST}:${UCB_PORT}"
echo "💾 Persistence file: ${UCB_PERSISTENCE_PATH}"
echo "🧠 Model weights: ${UCB_WEIGHTS_PATH}"
echo ""
echo "⚠️  IMPORTANT NOTES:"
echo "   - Keep this SINGLE-PROCESS (no --workers > 1)"
echo "   - UCB stats live in memory and sync across requests"
echo "   - Run locally (no VPS/VPN) for TopstepX compliance"
echo ""

# Start the service
python ucb_api.py
