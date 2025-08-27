#!/bin/bash

cd "$(dirname "$0")"

ENV_FILE=".env.local"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
    echo "Loaded environment variables from $ENV_FILE."
else
    echo "$ENV_FILE not found. Skipping env load."
fi

if [ "$BOT_QUICK_EXIT" = "1" ]; then
    echo "Quick-exit mode enabled (BOT_QUICK_EXIT=1). Orchestrator will start and exit after a short delay."
fi

SKIP_PROBE_RAW="$SKIP_CONNECTIVITY_PROBE"
SKIP_PROBE=false
if [ -n "$SKIP_PROBE_RAW" ]; then
    SKIP_PROBE_LOWER=$(echo "$SKIP_PROBE_RAW" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
    if [[ "$SKIP_PROBE_LOWER" =~ ^(1|true|yes)$ ]]; then
        SKIP_PROBE=true
    fi
fi

if [ "$SKIP_PROBE" = true ]; then
    echo "Skipping connectivity probe (SKIP_CONNECTIVITY_PROBE=$SKIP_PROBE_RAW)."
else
    echo "Running connectivity probe..."
    dotnet run --project ConnectivityProbe
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 1 ]; then
        echo "Connectivity probe failed (transport/network error). Fix connectivity or set SKIP_CONNECTIVITY_PROBE=true to bypass."
        exit $EXIT_CODE
    elif [ $EXIT_CODE -eq 2 ]; then
        echo "Connectivity probe: missing JWT/login credentials. Continuing to launch without blocking."
    else
        echo "Connectivity probe passed."
    fi
fi

if [ -z "$ASPNETCORE_URLS" ]; then
    export ASPNETCORE_URLS="http://localhost:5000"
fi
if [ -z "$APP_CONCISE_CONSOLE" ]; then
    export APP_CONCISE_CONSOLE="true"
fi

echo "Launching bot on $ASPNETCORE_URLS ..."
dotnet run --project src/OrchestratorAgent/OrchestratorAgent.csproj
echo "Bot process exited with code $?."
